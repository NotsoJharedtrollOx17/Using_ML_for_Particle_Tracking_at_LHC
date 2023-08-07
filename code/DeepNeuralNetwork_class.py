import torch
import torch.nn as nn
import torch.nn.functional as F


# TODO add initialization of class properties via __init__
# * Or you make another class for the three different arquitectures
# * Or you make it general and add the properties to the summary.json file in train.py
class DeepNeuralNetwork(nn.Module):
    # * Constructor
    def __init__(self):
        super(DeepNeuralNetwork, self).__init__()

        n_edge_features = 3
        n_node_features = 7
        input_size = 2 * n_node_features + n_edge_features # * 14
        n_hidden_layers = 2
        hidden_size = 100

        hidden_layers = []
        for layer_i in range(n_hidden_layers):
            hidden_layers.append(nn.Linear(hidden_size, hidden_size))
            hidden_layers.append(nn.ReLU())

        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            *hidden_layers,
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )

        #print(self)
    
    def forward(self, node_attr, edge_idxs, edge_attr):
        return self.layers(torch.cat((node_attr, edge_attr), dim=1)).unsqueeze(1)