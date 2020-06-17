import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
from collections import OrderedDict

from base import BaseModel
from .input_weighting import *


class LassoFFContinuous(BaseModel):
    """
    Data has dimension (N, T, M)
        - N samples
        - T dimensions for each feature
        - M features
    """
    def __init__(self, xdim, ydim, num_features, encoder_layers, decoder_layers, encoder_drop=0.0, decoder_drop=0.0):
        super().__init__()
        self.num_features = num_features
        self.in_dimension = xdim*num_features

        self.input_weights = InputWeighting(num_features)

        self.predictor = nn.Sequential(OrderedDict([
            ('diaglin0', nn.Linear(self.in_dimension, encoder_layers)),
            ('elu0', nn.ReLU()),
            ('drop0', nn.Dropout(encoder_drop)),
            ('diaglin0', nn.Linear(encoder_layers, ydim)),
        ]))


    def forward(self, x):
        return self.predictor(self.input_weights(x).view(x.shape[0],-1))
