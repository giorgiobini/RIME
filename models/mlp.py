import torch.nn as nn
import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import EMBEDDING_DIM


class MLP(nn.Module):
    def __init__(self, input_size, dividing_factor, num_hidden_layers, dropout_prob):
        super(MLP, self).__init__()
        
        num_layers = num_hidden_layers+2 #input and output layer

        first_layer_size = int(input_size/dividing_factor)

        hidden_sizes = list(np.linspace(2, first_layer_size, num_layers, dtype = np.int64)[1:][::-1])
        
        # Define the input layer
        self.input_layer = nn.Linear(input_size, first_layer_size)
        
        # Define the hidden layers
        self.hidden_layers = nn.ModuleList()
        for i in range(num_hidden_layers):
            self.hidden_layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
        
        # Define the output layer
        self.output_layer = nn.Linear(hidden_sizes[-1], 2)
        
        # Define the activation functions
        self.activation_functions = nn.ModuleList()
        self.activation_functions.append(nn.Tanh())
        for i in range(num_hidden_layers):
            self.activation_functions.append(nn.ReLU())
        
        # Define the dropout layers
        self.dropout_layers = nn.ModuleList()
        for i in range(num_hidden_layers+1): #+1 is for the first layer
            self.dropout_layers.append(nn.Dropout(p=dropout_prob))
        
    def forward(self, x):
        # Apply the input layer and the first activation function
        x = self.input_layer(x)
        x = self.dropout_layers[0](x)
        x = self.activation_functions[0](x)
        
        # Apply the hidden layers and activation functions
        for i in range(1, len(self.hidden_layers) + 1):
            x = self.hidden_layers[i - 1](x)
            x = self.dropout_layers[i](x)
            x = self.activation_functions[i](x)
        
        # Apply the output layer and activation function
        x = self.output_layer(x)
        return x
    
class MLP_fake(nn.Module):
    def __init__(self, input_size, dividing_factor, num_hidden_layers, dropout_prob):
        super(MLP_fake, self).__init__()
        
        # Linear layers with tanh activation function
        self.fc1 = nn.Linear(input_size, 30)
        self.fc2 = nn.Linear(30, 5)
        self.fc3 = nn.Linear(5, 5)
        self.fc4 = nn.Linear(5, 2)
        
        # Dropout layers
        self.dropout1 = nn.Dropout(p=dropout_prob)
        self.dropout2 = nn.Dropout(p=dropout_prob)
        self.dropout3 = nn.Dropout(p=dropout_prob)
        
        # Activation functions
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # Feedforward layers with dropout
        x = self.fc1(x)
        x = self.dropout1(x)
        x = self.relu(x)
        
        x = self.fc2(x)
        x = self.dropout2(x)
        x = self.relu(x)
        
        x = self.fc3(x)
        x = self.dropout3(x)
        x = self.relu(x)
        
        x = self.fc4(x)
        return x

    
def build(args):
    model = MLP(input_size=int(EMBEDDING_DIM*2), dividing_factor=args.dividing_factor, num_hidden_layers=args.num_hidden_layers, dropout_prob=args.dropout_prob)
    return model