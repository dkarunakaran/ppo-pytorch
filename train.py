import torch
import torch.optim as optim
import gymnasium as gym 
from actor import Actor
from critic import Critic
import yaml
from utility import logger_helper
from torch.distributions import Categorical
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

# Ref: the use of fabric - https://lightning.ai/docs/fabric/2.4.0/fundamentals/convert.html

class Train:
    def __init__(self):
        self.random_seed = 543
        self.env = gym.make('CartPole-v1')
        observation, info = self.env.reset()
        self.logger = logger_helper()
        with open("config.yaml") as f:
            self.cfg = yaml.load(f, Loader=yaml.FullLoader)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Device is: {self.device}")
        self.actor = Actor(self.env.observation_space.shape[0], self.env.action_space.n).to(self.device)
        self.critic = Critic(self.env.observation_space.shape[0]).to(self.device)
         

    def run(self):
        torch.manual_seed(self.cfg['train']['random_seed'])
        actor_optim = optim.Adam(self.actor.parameters(), lr=self.cfg['train']['lr'], betas=self.cfg['train']['betas'])
        critic_optim = optim.Adam(self.critic.parameters(), lr=self.cfg['train']['lr'], betas=self.cfg['train']['betas'])
        avg_actor_losses = []
        avg_critic_losses = []
        actor_losses = []
        critic_losses = []
        eps = np.finfo(np.float32).eps.item()
        batch_data = []
        for episode in range(self.cfg['train']['n_epidode']):
            rewards = []
            log_probs = []
            actions = []
            states = []
            #state_values = []
            self.actor.train()
            self.critic.train()
            terminated, truncated = False, False # initiate the terminated and truncated flags
            state, info = self.env.reset()
            # Converted to tensor
            state = torch.FloatTensor(state).to(self.device)
            self.logger.info(f"--------Episode: {episode} started----------")
            timesteps = 0
            # loop through timesteps
            #for i in range(self.cfg['train']['n_timesteps']):
            while not terminated and not truncated:
                timesteps += 1

                # The actor layer output the action probability as the actor NN has softmax in the output layer
                action_prob = self.actor(state)

                # As we know we do not use categorical cross entropy loss function directly, but contruct manually to have more control.
                # Categorical cross entropy loss function in pytorch does logits to probability using softmax to categorical distribution,
                # then compute the loss. So normally no need to add softmax function to the NN explicilty. In this work we add the softmax layer on the 
                # NN and compute the categorical distribution.
                
                # categorical function can  give categorical distribution from softmax probability or from logits(no softmax layer in output) with logits as attribute 
                action_dist= Categorical(action_prob)

                # Sample the action
                action = action_dist.sample()
                actions.append(action)

                # Get the log probability to get log pi_theta(a|s) and save it to a list.
                log_probs.append(action_dist.log_prob(action))

                # Compute the current state-value 
                #v_st = self.critic(state)
                #state_values.append(v_st)

                states.append(state)

                # Action has to convert from tensor to numpy for env to process
                next_state, reward, terminated, truncated, info = self.env.step(action.item())
                rewards.append(reward)

                # Assign next state as current state
                state = torch.FloatTensor(next_state).to(self.device)

                # Enviornment return done == true if the current episode is terminated
                if terminated or truncated:
                    self.logger.info('Iteration: {}, Score: {}'.format(episode, timesteps))
                    break

            R = 0
            returns = [] # list to save the true values

            # Calculate the return of each episode using rewards returned from the environment in the episode
            for r in rewards[::-1]:
                # Calculate the discounted value
                R = r + self.cfg['train']['gamma'] * R
                returns.insert(0, R)

            returns = torch.tensor(returns).to(self.device)
            returns = (returns - returns.mean()) / (returns.std() + eps)

            # Store the data
            batch_data.append([states, actions, returns, log_probs])

            if episode != 0 and episode%self.cfg['train']['update_freq'] == 0:
                # This is the loop where we update our network for some n epochs.This additonal for loops
                # improved the training
                for _ in range(5):
                    for states_b, actions_b, returns_b, old_log_probs in batch_data:
                        
                        # Convert list to tensor
                        old_states = torch.stack(states_b, dim=0).detach()
                        old_actions = torch.stack(actions_b, dim=0).detach()
                        old_log_probs = torch.stack(old_log_probs, dim=0).detach()

                        state_values = self.critic(old_states)

                        # calculate advantages
                        advantages = returns_b.detach() - state_values.detach()

                        # Ref: https://github.com/ericyangyu/PPO-for-Beginners/blob/master/ppo.py
                        # Normalizing advantages isn't theoretically necessary, but in practice it decreases the variance of 
                        # our advantages and makes convergence much more stable and faster. I added this because
                        # solving some environments was too unstable without it.
                        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-10)

                        # Now we need to compute the ratio (pi_theta / pi_theta__old). In order to do that, we need to get old policy of the 
                        # action taken from the state which is stored and compute the new policy of the same action
                        # The actor layer output the action probability as the actor NN has softmax in the output layer
                        action_probs = self.actor(old_states)
                        dist = Categorical(action_probs)
                        new_log_probs = dist.log_prob(old_actions)
                        # Because we are taking log, we can substract instead division. Then taking the exponents will give the same result as division
                        ratios = torch.exp(new_log_probs - old_log_probs)
                        
                        # Unclipped part of the surrogate loss function
                        surr1 = ratios * advantages

                        # Clipped part of the surrogate loss function
                        surr2 = torch.clamp(ratios, 1 - self.cfg['train']['clip_param'], 1 + self.cfg['train']['clip_param']) * advantages

                        # Update actor network: loss = min(surr1, surr2)
                        actor_loss = -torch.min(surr1, surr2).mean()
                        actor_losses.append(actor_loss.item())
                        
                        # Calculate critic (value) loss using huber loss
                        # Huber loss, which is less sensitive to outliers in data than squared-error loss. In value based RL ssetup, huber loss is preferred.
                        # Smooth L1 loss is closely related to HuberLoss
                        critic_loss =  F.smooth_l1_loss(state_values, returns_b.unsqueeze(1)) #F.huber_loss(state_value, torch.tensor([R]))
                        critic_losses.append(critic_loss.item())

                        actor_optim.zero_grad()
                        critic_optim.zero_grad()

                        # Perform backprop
                        actor_loss.backward()
                        critic_loss.backward()
                        
                        # Perform optimization
                        actor_optim.step()
                        critic_optim.step()

                # Clear the data
                batch_data = []

            # Storing average losses for plotting
            if episode%self.cfg['train']['store_plotting_data'] == 0:
                avg_actor_losses.append(np.mean(actor_losses))
                avg_critic_losses.append(np.mean(critic_losses))
                actor_losses = []
                critic_losses = []
        
        plt.figure(figsize=(10,6))
        plt.xlabel("X-axis")  # add X-axis label
        plt.ylabel("Y-axis")  # add Y-axis label
        plt.title("Average actor loss")  # add title
        plt.plot(avg_actor_losses)
        plt.savefig('actor_loss.png')
        plt.close()

        plt.figure(figsize=(10,6))
        plt.xlabel("X-axis")  # add X-axis label
        plt.ylabel("Y-axis")  # add Y-axis label
        plt.title("Average critic loss")  # add title
        plt.plot(avg_critic_losses)
        plt.savefig('critic_loss.png')
        plt.close()

        torch.save(self.actor, 'model/actor.pkl')
        torch.save(self.critic, 'model/critic.pkl')
        self.env.close()



if __name__ == '__main__':
    train = Train()
    train.run()