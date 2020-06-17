import torch
import torch.nn as nn
from torch.autograd import Variable


class BlockDropout(nn.Module):
    """
    Inverted dropout, except drops out entire dimensions with same probability
    """
    def __init__(self, p_drop, force_nonzero=True):
        super().__init__()
        self.p_drop = p_drop
        self.force_nonzero = force_nonzero


    def forward(self, z, enabled=True):
        # input dimension: (N, dim, n_features)
        # choose random features to drop out entirely
        if enabled:
        # if self.training and enabled:
            binomial = torch.distributions.binomial.Binomial(probs=1-self.p_drop)
            mask = Variable(z.new(z.shape))
            mask[:,0,:] = binomial.sample((z.shape[0], z.shape[2]))

            # force at least one to be nonzero
            if self.force_nonzero:
                for i,idx in enumerate(torch.multinomial(torch.ones(z.shape[2]), z.shape[0], replacement=True)):
                    if torch.all(mask[i,0,:] == 0):
                        mask[i,0,idx] = 1

            for j in range(1, z.shape[1]):
                mask[:,j,:] = mask[:,0,:]
            
            return mask*z

        else:
            return (1-self.p_drop)*z
