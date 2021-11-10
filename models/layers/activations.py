import torch
import torch.nn as nn
import torch.nn.functional as F

def adaptive_elu(x, xshift, yshift):
    return F.elu(x - xshift, inplace=True) + yshift

class AdaptiveELU(nn.Module):
    """
    ELU shifted by user specified values. This helps to ensure the output to stay positive.
    """

    def __init__(self, xshift, yshift, **kwargs):
        super(AdaptiveELU, self).__init__(**kwargs)

        self.xshift = xshift
        self.yshift = yshift

    def forward(self, x):
        return adaptive_elu(x, self.xshift, self.yshift)

NLtypes = {
    'none': None,
    'lin': None,
    'relu': nn.ReLU(),
    'elu': AdaptiveELU(0.0, 1.0),
    'square': torch.square, # this doesn't exist: just apply exponent?
    'softplus': nn.Softplus(),
    'tanh': nn.Tanh(),
    'sigmoid': nn.Sigmoid()
    }