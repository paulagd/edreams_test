import torch
import torch.nn as nn
import torch.nn.functional as F
from IPython import embed


class NeuralNetwork(torch.nn.Module):
    """
    Deep Learning model
    """
    def __init__(self, dropout, input_dim, output_dim=1):
        # super function. It inherits from nn.Module and we can access everythink in nn.Module
        super(NeuralNetwork, self).__init__()
        # Linear function.
        # self.embedding = nn.
        self.input = nn.Linear(input_dim, input_dim*2)
        self.hidden = nn.Linear(input_dim*2, input_dim)
        self.out = nn.Linear(input_dim, output_dim)

    def forward(self, x):

        x = F.relu(self.input(x))
        x = F.relu(self.hidden(x))
        x = self.out(x)
        return torch.sigmoid(x)
        # return F.softmax(x)