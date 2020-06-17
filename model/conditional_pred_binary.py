import torch
import torch.nn as nn
from torch.autograd import Variable


class InputWeighting(nn.Module):
    """
    Inverted dropout, except drops out entire dimensions with same probability
    """
    def __init__(self, num_features):
        super().__init__()
        self.num_features = num_features
        self.weight = nn.Parameter(torch.randn(num_features))

        # weight initialization
        for param in self.parameters():
            if param.dim() < 2:
                nn.init.constant_(param, 0)
            else:
                torch.nn.init.xavier_uniform_(param)


    def forward(self, x):
        # input dimension: (N, dim, n_features)
        return x * self.weight
