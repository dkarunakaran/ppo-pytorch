import torch
import torch.nn as nn
from torch.distributions import Categorical

# Ref: https://github.com/yc930401/Actor-Critic-pytorch

class Actor(nn.Module):
    def __init__(self, state_size=None, action_size=None):
        super().__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.linear1 = nn.Linear(self.state_size, 128)
        self.linear2 = nn.Linear(128, 256)
        self.linear3 = nn.Linear(256, self.action_size)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()

    
    def forward(self, x):
        # x is state
        output = self.relu(self.linear1(x))
        output = self.relu(self.linear2(output))
        output = self.linear3(output)
        # The loss function of this NN will be categorical cross entropy. But in normal pytorch, we do no need to
        # explicty compute the softmax as it will be part of the categorical cross entropy loss in pytorch. But here we aply the 
        # categorical cross entropy loss manually.
        return self.softmax(output)
    

if __name__ == '__main__':
    actor = Actor(10,2)
    print(actor)