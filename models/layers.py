import math
import numpy as np
import torch
from torch import nn
from typing import Tuple, Union

from torch.nn import functional as F
from torch.nn.parameter import Parameter
from torch.nn.common_types import _size_2_t # for conv2,conv3 default
from torch.nn import init
from torch.nn.parameter import Parameter

"""
1. posConv2D:   2D convolutional with positive weights
2. PosLinear:   Linear layer with positive weights
3. AdaptiveELU: adaptive exponential linear unit
4. EiMask:      apply excitation/inhibition mask on outputs

"""
class posConv2D(nn.Conv2d):
    """
    2D convolutional layer that is constrained to have positive weights
    """
    def __init__(self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: _size_2_t = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros'
    ):

        super(posConv2D, self).__init__(in_channels,
            out_channels, kernel_size, stride,
            padding, dilation, groups, bias, padding_mode)
        self.register_buffer("minval", torch.tensor(0.0))

    def forward(self, x):
        posweight = torch.maximum(self.weight, self.minval)
        return self._conv_forward(x, posweight)


class PosLinear(nn.Linear):
    """
    Linear layer with constrained positive weights
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super(PosLinear, self).__init__(in_features=in_features, out_features=out_features, bias=bias)
        self.register_buffer("minval", torch.tensor(0.0))

    def forward(self, x):
        pos_weight = torch.maximum(self.weight, self.minval)
        return F.linear(x, pos_weight, self.bias)


class ShapeLinear(nn.Module):
    """
    Linear layer with true shape to weights
    
    Input is flattened and then it works like a normal linear layer. This only helps to do regularization.

    """
    def __init__(self, in_features, out_features:int, bias: bool = True, positive: bool = False):
        super(ShapeLinear, self).__init__()

        self.in_features = in_features
        # self.flatten = nn.Flatten()
        
        self.shape = tuple([out_features] + list(in_features))
        self.positive_constraint = positive

        self.weight = Parameter(torch.Tensor( size =self.shape ))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_buffer('bias', torch.zeros(out_features))

        if self.positive_constraint:
            self.register_buffer("minval", torch.tensor(0.0))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)          

    def forward(self, x):
        w = self.weight
        if self.positive_constraint:
            w = torch.maximum(w, self.minval)
        x = torch.einsum('ncwh,kcwh->nk', x, w)
        return x + self.bias
        # if self.bias:
        # x = x + self.bias
        



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

class EiMask(nn.Module):
    """
        Apply signed mask. Should work regardless of whether it's a linear or convolution model.
        Uses einstein summation.
    """
    def __init__(self, ni, ne):
        """
        ni: number of inhibitory units
        ne: number of excitatory units
        """
        super(EiMask,self).__init__()

        self.register_buffer("ei_mask", torch.cat((torch.ones((1,ne)), -torch.ones((1,ni))), axis=1).squeeze())
    
    def forward(self, x):
        out = torch.einsum("nc...,c->nc...", x, self.ei_mask)
        return out

class divNorm(nn.Module):
    """
    Divisive Normalization layer

    """
    def __init__(self, in_features):
        super(divNorm, self).__init__()
        
        self.in_features = in_features
        self.weight = Parameter(torch.Tensor(in_features, in_features))
        self.bias = Parameter(torch.Tensor(in_features))

        self.relu = F.relu
        self.reset_parameters()

    def reset_parameters(self) -> None:
        print("divNorm: initialize weights custom")
        self.weight.data[:] = 1.0
        self.bias.data[:] = .5
        # nn.init.uniform_(self.weight, 0.0, 1.0)
        # nn.init.uniform_(self.bias, 0.0, 1.0)

    def forward(self, x):

        posweight = self.relu(self.weight)
        x = self.relu(x)
        xdiv = torch.einsum('nc...,ck->nk...', x, posweight)
        if len(x.shape)==4:
            xdiv = xdiv + self.bias[None,:,None,None] # is convolutional
        else:
            xdiv = xdiv + self.bias[None,:]

        x = x / xdiv.clamp_(0.001) # divide

        return x

class STconv(nn.Module):
    """
    Spatiotemporal convolution layer
    This layer takes in a spatial input and applies a convolution along the batch dimension as if it were time
    This assumes the batch dimension is temporally contiguous and produces invalid samples at the begining, but it
    has a smaller memory footprint.
    """
    def __init__(self, input_dims,
        kernel_size = (1,10,10,10), # (C, T, W, H)
        out_features = 10,
        tent_spacing=None,
        stride=1,
        dilation=1,
        bias=None,
        positive_constraint=False):

        super(STconv, self).__init__()
        
        # dims = [C x H x W ]
        assert len(input_dims) == 3, "STconv: input must be 3D [C x H x W]. Use 1 for empty dims. Time gets created within."
        self.input_dims = list(input_dims) + [1] # add time dim at end
        self.num_lags = kernel_size[1]
        self.kernel_size = kernel_size
        self.shape = [out_features] + list(kernel_size)
        self.positive_constraint = positive_constraint

        if tent_spacing is not None:
            from neureye.models.utils import tent_basis_generate
            tent_basis = tent_basis_generate(np.arange(0, self.num_lags, tent_spacing))/tent_spacing
            num_lag_params = tent_basis.shape[1]
            self.register_buffer('tent_basis', torch.Tensor(tent_basis.T))
            self.shape[2] = num_lag_params
        else:
            self.tent_basis = None

        self.stride = stride
        self.dilation = dilation

        # weight needs to be NDIMS x out_features
        self.weight = Parameter(torch.Tensor( size = self.shape ))

        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.bias = None #self.register_buffer('bias', torch.zeros(out_features))

        if self.positive_constraint:
            self.register_buffer("minval", torch.tensor(0.0))
            self.relu = F.relu

        # Do spatial padding manually (TODO: double check this)
        self.padding = (self.shape[3]//2, (self.shape[3] - 1 + self.shape[3]%2)//2,
            self.shape[4]//2, (self.shape[4] - 1 + self.shape[4]%2)//2,
            self.num_lags-1-(1-tent_spacing%2), 0)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=5**.5)
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            # bound = 1 / math.sqrt(fan_in)
            bound = fan_in**-.5
            init.uniform_(self.bias, -bound, bound)  

    def forward(self, x):
        # Reshape stim matrix LACKING temporal dimension [bcwh] 
        # and inputs (note uses 4-d rep of tensor before combinine dims 0,3)
        # pytorch likes 3D convolutions to be [B,C,T,W,H].
        # I benchmarked this and it'sd a 20% speedup to put the "Time" dimension first.

        w = self.weight
        if self.positive_constraint:
            w = torch.maximum(w, self.minval)

        s = x.reshape([-1] + self.input_dims).permute(4,1,0,2,3) # [1,C,B,W,H]
        # w = w.reshape(self.filter_dims+[-1]).permute(4,0,3,1,2) # [N,C,T,W,H] 
        
        # time-expand using tent-basis if it exists
        if self.tent_basis is not None:
            w = torch.einsum('nctwh,tz->nczwh', w, self.tent_basis)
        y = F.conv3d(
            F.pad(s, self.padding, "constant", 0),
            w, 
            bias=self.bias,
            stride=self.stride, dilation=self.dilation)
        
        y = y.reshape(y.shape[1:]).permute(1,0,2,3)
        
        return y