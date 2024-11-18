import torch
import torch.nn as nn
from torch.distributions import Categorical

# Ref: https://github.com/yc930401/Actor-Critic-pytorch

class Critic(nn.Module):
    def __init__(self, state_size=None):
        super().__init__()
        self.state_size = state_size
        self.linear1 = nn.Linear(self.state_size, 128)
        self.linear2 = nn.Linear(128, 256)
        self.linear3 = nn.Linear(256, 1)
        self.relu = nn.ReLU()

    
    def forward(self, x):
        # x is state
        output = self.relu(self.linear1(x))
        output = self.relu(self.linear2(output))
        return self.linear3(output)
    

if __name__ == '__main__':
    critic = Critic(10)
    print(critic)