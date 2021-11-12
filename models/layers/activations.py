import torch
import torch.nn as nn
import torch.nn.functional as F

def adaptive_elu(x, xshift, yshift, inplace=False):
    return F.elu(x - xshift, inplace) + yshift

class AdaptiveELU(nn.Module):
    """
    Exponential Linear Unit shifted by user specified values.
    This helps to ensure the output to stay positive.
    """

    def __init__(self, xshift=0.0, yshift=1.0, inplace=False, **kwargs):
        super(AdaptiveELU, self).__init__(**kwargs)

        self.xshift = xshift
        self.yshift = yshift
        self.inplace = inplace

    def forward(self, x):
        return adaptive_elu(x, self.xshift, self.yshift, self.inplace)

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