import torch
import torch.nn as nn
from torch.autograd import Variable


class CondPredContinuous(nn.Module):
    def __init__(self, num_features, layers1, layers2, dropout):
        super().__init__()
        self.num_features = num_features


    def forward(self, z, w):
        # z shape: (n,d,m), w shape: (n,m)
        concat = torch.cat((z, w.view(w.shape[0],1,-1)), dim=1)
