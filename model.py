import torch
import torch.nn as nn

class Actor(nn.Module):
    def __init__(self, states = 33, actions = 4, layer1 = 256, layer2=256):
        super().__init__()
        self.L1 = nn.Linear(states, layer1)
        self.L2 = nn.Linear(layer1, layer2)
        self.mean = nn.Linear(layer2, actions) 
        self.std = nn.Linear(layer2, actions)
        # Probably wont need dropout 
#         self.dropout = nn.Dropout(p = 0.1)
        self.activation = nn.ReLU()
        
    
    ## implementation of forward pass 
    def forward(self, states):
        out =  self.activation(self.L1(states))
        out =  self.activation(self.L2(out))
        out_mean = self.mean(out)
        out_std = self.std(out)
        out_std = torch.clamp(out_std, -20, 2)
        # Return sampling parameters
        return out_mean, out_std
        
class tdActor(nn.Module):
    def __init__(self, states = 33, actions = 4, layer1 = 256, layer2=256):
        super().__init__()
        self.L1 = nn.Linear(states, layer1)
        self.L2 = nn.Linear(layer1, layer2)
        self.out = nn.Linear(layer2, actions) 
 
        # Probably wont need dropout 
#         self.dropout = nn.Dropout(p = 0.1)
        self.activation = nn.ReLU()
         
    ## implementation of forward pass 
    def forward(self, states):
        out =  self.activation(self.L1(states))
        out =  self.activation(self.L2(out))
        out = self.out(out)
        return torch.tanh(out)
    
class Critic(nn.Module):
    def __init__(self, states = 33, actions = 4, layer1 = 256, layer2 = 256):
        super().__init__()
        
        self.L1 = nn.Linear(states+actions, layer1)
        self.L2 = nn.Linear(layer1, layer2)
        self.out = nn.Linear(layer2, 1)
        self.activation = nn.ReLU()
        
    def forward(self, states, actions):
#         print(states.size(), actions.size())
        inputs = torch.cat([states, actions], 1)
        out =  self.activation(self.L1(inputs))
        out =  self.activation(self.L2(out))
        out = self.out(out)
        return out