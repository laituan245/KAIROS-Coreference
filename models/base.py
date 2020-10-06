import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from transformers import *
from math import ceil, floor

# BaseModel
class BaseModel(nn.Module):
    def __init__(self, configs):
        super(BaseModel, self).__init__()
        self.configs = configs
        self.device = torch.device('cuda' if torch.cuda.is_available() and not configs['no_cuda'] else 'cpu')

# FFNN Module
class FFNNModule(nn.Module):
    """ Generic FFNN-based Scoring Module
    """
    def __init__(self, input_size, hidden_sizes, output_size=1, dropout = 0.2):
        super(FFNNModule, self).__init__()
        self.layers = []

        prev_size = input_size
        for hidden_size in hidden_sizes:
            self.layers.append(nn.Linear(prev_size, hidden_size))
            self.layers.append(nn.ReLU(True))
            self.layers.append(nn.Dropout(dropout))
            prev_size = hidden_size

        self.layers.append(nn.Linear(prev_size, output_size))

        self.layer_module = nn.ModuleList(self.layers)

    def forward(self, x):
        out = x
        for layer in self.layer_module:
            out = layer(out)
        return out.squeeze()
