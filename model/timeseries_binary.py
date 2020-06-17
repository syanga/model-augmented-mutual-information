import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
from collections import OrderedDict

from base import BaseModel
from .base_features import FeaturesBase

from .diagonal_linear import *
from .diagonal_rnn import *

        
class TimeSeriesParallel(FeaturesBase):
    """
    Data has dimension (N, T, M)
        - N samples
        - M features
        - T dimensions for each feature

    Embedding has dimension (N, dim_z_per, num_features=M)

    self.encoders is module list, e.g.
        # self.encoders = nn.ModuleList([
        #     nn.Linear(xdim, zdim)
        #     for i in range(num_features)
        # ])
    """
    def __init__(self, num_features, dim_z_per, 
                 rnn_states, rnn_layers, decoder_layers, 
                 rnn_type='GRU', block_drop=0.0, rnn_drop=0.0, decoder_drop=0.0, z_offest=2.0):
        super().__init__(None, 1, num_features, dim_z_per, block_drop)

        if rnn_type == "GRU":
            rnn_class = DiagonalGRU
        elif rnn_type == "RNN":
            rnn_class = DiagonalRNN
        elif rnn_type == "LSTM":
            rnn_class = DiagonalLSTM

        self.z_offest = z_offest

        # x -> z
        self.encoders = nn.ModuleList([
            rnn_class(1, rnn_states, num_layers=rnn_layers, dropout=rnn_drop, batch_first=True, n_copies=num_features, shared_copies=False),
            DiagonalLinear(num_features, rnn_states, dim_z_per)
        ])

        # z -> y
        self.decoder = nn.Sequential(OrderedDict([
            # ('elu0', nn.ELU()),
            ('ffnn0', nn.Linear(self.dim_z_total, decoder_layers)),
            ('elu1', nn.ELU()),
            ('drop0', nn.Dropout(decoder_drop)),
            ('ffnn1', nn.Linear(decoder_layers, 1)),
            ('sigmoid0', nn.Sigmoid())
        ]))


    def _encode(self, x):
        y,h = self.encoders[0](x)
        return self.encoders[1](y[:,-1,:]).view(x.shape[0], self.num_features, self.dim_z_per).permute(0,2,1).contiguous()# + self.z_offest


    def _predict(self, z, apply_block_drop=True):
        return self.decoder(self.block_drop(z, enabled=apply_block_drop).contiguous().view(z.shape[0],-1))
        

    def divergence(self, z1, z2, eps=1e-5):
        p = self._predict(z1, apply_block_drop=False).view(-1).clamp(min=eps, max=1.0-eps)
        q = self._predict(z2, apply_block_drop=False).view(-1).clamp(min=eps, max=1.0-eps)
        return 0.5 * (p-q) * (p.log() - q.log() - (1-p).log() + (1-q).log())


    def forward(self, x, force_block_drop=True):
        """
        Training pipeline: x->z->y_pred
        Return also the embedding z
        """
        z = self._encode(x)
        return self._predict(z), z



class TimeContinuousParallel(FeaturesBase):
    """
    Data has dimension (N, T, M)
        - N samples
        - M features
        - T dimensions for each feature

    Embedding has dimension (N, dim_z_per, num_features=M)

    self.encoders is module list, e.g.
        # self.encoders = nn.ModuleList([
        #     nn.Linear(xdim, zdim)
        #     for i in range(num_features)
        # ])
    """
    def __init__(self, num_features, dim_z_per, 
                 rnn_states, rnn_layers, decoder_layers, 
                 rnn_type='GRU', block_drop=0.0, rnn_drop=0.0, decoder_drop=0.0):
        super().__init__(None, 1, num_features, dim_z_per, block_drop)

        if rnn_type == "GRU":
            rnn_class = DiagonalGRU
        elif rnn_type == "RNN":
            rnn_class = DiagonalRNN
        elif rnn_type == "LSTM":
            rnn_class = DiagonalLSTM

        # x -> z
        self.encoders = nn.ModuleList([
            rnn_class(1, rnn_states, num_layers=rnn_layers, dropout=rnn_drop, batch_first=True, n_copies=num_features, shared_copies=False),
            DiagonalLinear(num_features, rnn_states, dim_z_per)
        ])

        # z -> y
        self.decoder = nn.Sequential(OrderedDict([
            # ('elu0', nn.ELU()),
            ('ffnn0', nn.Linear(self.dim_z_total, decoder_layers)),
            ('elu1', nn.ELU()),
            ('drop0', nn.Dropout(decoder_drop)),
            ('ffnn1', nn.Linear(decoder_layers, 1)),
        ]))


    def _encode(self, x):
        y,h = self.encoders[0](x)
        return self.encoders[1](y[:,-1,:]).view(x.shape[0], self.num_features, self.dim_z_per).permute(0,2,1).contiguous()


    def _predict(self, z, apply_block_drop=True):
        return self.decoder(self.block_drop(z, enabled=apply_block_drop).contiguous().view(z.shape[0],-1))


    def divergence(self, z1, z2):
        y1 = self._predict(z1, apply_block_drop=False)
        y2 = self._predict(z2, apply_block_drop=False)
        return torch.sum(((y1-y2)**2).view(y1.shape[0], -1), dim=1)


    def forward(self, x, force_block_drop=True):
        """
        Training pipeline: x->z->y_pred
        Return also the embedding z
        """
        z = self._encode(x)
        return self._predict(z), z






# class TimeSharedParallel(FeaturesBase):
#     """
#     Data has dimension (N, T, M)
#         - N samples
#         - M features
#         - T dimensions for each feature

#     Embedding has dimension (N, dim_z_per, num_features=M)

#     self.encoders is module list, e.g.
#         # self.encoders = nn.ModuleList([
#         #     nn.Linear(xdim, zdim)
#         #     for i in range(num_features)
#         # ])
#     """
#     def __init__(self, num_features, dim_z_per, 
#                  shared_states, shared_layers, 
#                  rnn_states, rnn_layers, decoder_layers, 
#                  rnn_type='GRU', block_drop=0.0, shared_drop=0.0, rnn_drop=0.0, decoder_drop=0.0):
#         super().__init__(None, 1, num_features, dim_z_per, block_drop)

#         # x -> z
#         self.shared_rnn = DiagonalGRU(1, shared_states, num_layers=shared_layers, batch_first=True, n_copies=num_features, shared_copies=True)
#         self.encoder_rnn = DiagonalGRU(shared_states, rnn_states, num_layers=rnn_layers, batch_first=True, n_copies=num_features, shared_copies=False)
#         self.encoder_linear = DiagonalLinear(num_features, rnn_states, dim_z_per)

#         # z -> y
#         self.decoder = nn.Sequential(OrderedDict([
#             ('ffnn0', nn.Linear(self.dim_z_total, decoder_layers)),
#             ('elu0', nn.ELU()),
#             ('drop0', nn.Dropout(decoder_drop)),
#             ('ffnn1', nn.Linear(decoder_layers, 1)),
#             ('sigmoid0', nn.Sigmoid())
#         ]))


#     def _encode(self, x):
#         y,h = self.shared_rnn(x)
#         y,h = self.encoder_rnn(y)
#         z = self.encoder_linear(y[:,-1,:]).view(x.shape[0], self.num_features, self.dim_z_per).permute(0,2,1).contiguous()
#         return z


#     def _predict(self, z):    
#         # return self.decoder(self.block_drop(z).view(z.shape[0], -1))
#         return self.decoder(z.view(z.shape[0], -1))
        

#     def forward(self, x):
#         """
#         Training pipeline: x->z->y_pred
#         Return also the embedding z
#         """
#         z = self.block_drop(self._encode(x))
#         return self._predict(z), z
