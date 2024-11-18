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
        with open("config.yaml") as f:
            self.cfg = yaml.load(f, Loader=yaml.FullLoader)
        
        self.logger = logger_helper()
        self.actor = Actor(self.env.observation_space.shape[0], self.env.action_space.n)
        self.critic = Critic(self.env.observation_space.shape[0])
        

    def run(self):
        torch.manual_seed(self.cfg['train']['random_seed'])
        actor_optim = optim.Adam(self.actor.parameters(), lr=self.cfg['train']['lr'], betas=self.cfg['train']['betas'])
        critic_optim = optim.Adam(self.critic.parameters(), lr=self.cfg['train']['lr'], betas=self.cfg['train']['betas'])
        actor_losses = []
        avg_actor_losses = []
        critic_losses = []
        avg_critic_losses = []
        eps = np.finfo(np.float32).eps.item()
        for episode in range(self.cfg['train']['n_epidode']):
            rewards = []
            log_probs = []
            actions = []
            states = []

            state = self.env.reset()
            # Converted to tensor
            state = torch.FloatTensor(state[0])
            self.logger.info(f"--------Episode: {episode} started----------")
            actor_optim.zero_grad()
            critic_optim.zero_grad()
            # loop through timesteps
            for i in range(self.cfg['train']['n_timesteps']):
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

                states.append(state)

                # Action has to convert from tensor to numpy for env to process
                next_state, reward, done, _, _= self.env.step(action.detach().numpy())
                rewards.append(reward)

                # Assign next state as current state
                state = torch.FloatTensor(next_state) 

                # Enviornment return done == true if the current episode is terminated
                if done:
                    self.logger.info('Iteration: {}, Score: {}'.format(episode, i))
                    break

            R = 0
            actor_loss_list = [] # list to save actor (policy) loss
            critic_loss_list = [] # list to save critic (value) loss
            returns = [] # list to save the true values

            # Calculate the return of each episode using rewards returned from the environment in the episode
            for r in rewards[::-1]:
                # Calculate the discounted value
                R = r + self.cfg['train']['gamma'] * R
                returns.insert(0, R)

            returns = torch.tensor(returns)
            returns = (returns - returns.mean()) / (returns.std() + eps)
            
            # Optimize the model
            for old_logprobs, state, action, R in zip(log_probs, states, actions, returns):
                
                # Compute the current state-value v_s_t
                state_value = self.critic(state)

                # Advantage is calculated by the difference between actual return of current stateand estimated return of current state(v_st)
                advantage = torch.FloatTensor(R - state_value.item())

                # Now we need to compute the ratio (pi_theta / pi_theta__old). In order to do that, we need to get old policy of the 
                # action taken from the state which is stored and compute the new policy of the same action
                # The actor layer output the action probability as the actor NN has softmax in the output layer
                action_prob = self.actor(state)
                action_dist= Categorical(action_prob)
                new_logprob = action_dist.log_prob(action)

                # Because we are taking log, we can substract instead division. Then taking the exponents will give the same result as division
                ratio = torch.exp(new_logprob - old_logprobs)
                
                # Unclipped part of the surrogate loss function
                surr1 = ratio * advantage

                # Clipped part of the surrogate loss function
                surr2 = torch.clamp(ratio, 1 - self.cfg['train']['clip_param'], 1 + self.cfg['train']['clip_param']) * advantage

                # Update actor network: loss = min(surr1, surr2)
                a_loss = -torch.min(surr1, surr2).mean()
                actor_loss_list.append(a_loss)

                # Calculate critic (value) loss using huber loss
                # Huber loss, which is less sensitive to outliers in data than squared-error loss. In value based RL ssetup, huber loss is preferred.
                # Smooth L1 loss is closely related to HuberLoss
                c_loss =  F.smooth_l1_loss(state_value, torch.tensor([R])) #F.huber_loss(state_value, torch.tensor([R]))
                critic_loss_list.append(c_loss)

            # Sum up all the values of actor_losses(policy_losses) and critic_loss(value_losses)
            actor_loss = torch.stack(actor_loss_list).sum()
            critic_loss = torch.stack(critic_loss_list).sum()

            # Perform backprop
            actor_loss.backward()
            critic_loss.backward()
            
            # Perform optimization
            actor_optim.step()
            critic_optim.step()

            # Storing average losses for plotting
            if episode%50 == 0:
                avg_actor_losses.append(np.mean(actor_losses))
                avg_critic_losses.append(np.mean(critic_losses))
                actor_losses = []
                critic_losses = []
            else:
                actor_losses.append(actor_loss.detach().numpy())
                critic_losses.append((critic_loss.detach().numpy()))

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