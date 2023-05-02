import torch
import torch.nn as nn
from torch.nn import Linear
import torch.nn.functional as F

"""
Neural Network architecture
Weight Initiatialization is done using xavier_uniform.
Bias initialization is done using normal distribution.
Every hidden layer has ReLU activation function.
I have chosen only 3 layers because 4 and 5 werent performing any better.
"""
class Net3 (nn.Module):
    def __init__(self, feature, hidden, output):
        super(Net3, self).__init__()        

        self.hidden1 = Linear(feature, hidden)    
        nn.init.xavier_uniform_(self.hidden1.weight, gain=nn.init.calculate_gain('relu',0.2))
        nn.init.normal_(self.hidden1.bias, mean=0.25, std=1.0)
        
        self.hidden2 = Linear(hidden, hidden)  
        nn.init.xavier_uniform_(self.hidden2.weight, gain=nn.init.calculate_gain('relu',0.2))
        nn.init.normal_(self.hidden2.bias, mean=0.25, std=1.0)        
        
        self.hidden3 = Linear(hidden, hidden)      
        nn.init.xavier_uniform_(self.hidden3.weight, gain=nn.init.calculate_gain('relu',0.2))
        nn.init.normal_(self.hidden3.bias,mean=0.25, std=1.0)               
        
        self.out = Linear(hidden, output)              
    
    def forward(self, x):        
        x = torch.relu(self.hidden1(x))      
        x = torch.relu(self.hidden2(x))      
        x = torch.relu(self.hidden3(x))         
        x = self.out (x)            
        return x
    

