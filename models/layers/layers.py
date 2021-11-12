import math
import numpy as np
import torch
from torch import nn
# from typing import Tuple, Union

from torch.nn import functional as F
from torch.nn.parameter import Parameter
from torch.nn.common_types import _size_2_t # for conv2,conv3 default
from torch.nn import init
from torch.nn.parameter import Parameter

from neureye.models.regularizers import Regularization
from .activations import *

from copy import deepcopy

'''
1. NDNLayer (base class, linear layer)
2. ConvLayer (convolutional NDNLayer, can be 1D or 2D)
3. STConvLayer (Spatio-Temporal Convolutional NDNLayer, uses the batch dimension as contiguous time)
4. DivNormLayer (Divisive Normalization Layer, can operate on linear or conv layers, has no spatial normalization pool)
5. FixationLayer 

'''

class NDNLayer(nn.Module):
    """
    Base class for NDN layers.
    Handles the weight initialization and regularization

    Args:
        input_dims: list of 4 ints, dimensions of input
            [Channels, Height, Width, Lags]
            use 1 if the dimension is not used
        num_filters: int, number of filters in layer
        NLtype: str, type of nonlinearity to use (see activations.py)
        norm: int, type of filter normalization to use (0=None, 1=filters are unit vectors, 2=maxnorm?)
        pos_constraint: bool, whether to constrain weights to be positive
        num_inh: int, number of inhibitory units (creates ei_mask and makes "inhibitory" units have negative output)
        bias: bool, whether to include bias term
        weight_init: str, type of weight initialization to use (see reset_parameters, default 'xavier_uniform')
        bias_init: str, type of bias initialization to use (see reset_parameters, default 'zeros')
        reg_vals: dict, regularization values to use (see regularizers.py)

    NDNLayers have parameters 'weight' and 'bias' (if bias=True)

    The forward has steps:
        1. preprocess_weights
        2. main computation (this is just a linear layer for the base class)
        3. nonlinearity
        4. ei_mask
    
    weight is always flattened to a vector, and then reshaped to the appropriate size
    use get_weights() to get the weights in the correct shape
    
    preprocess_weights() applies positive constraint and normalization if requested
    compute_reg_loss() computes the regularization loss for the regularization specified in reg_vals

    """
    def __init__(self, input_dims=None,
            num_filters=None,
            filter_dims=None,
            NLtype:str='lin',
            norm_type:int=0,
            pos_constraint=False,
            num_inh:int=0,
            bias:bool=False,
            weights_initializer:str='xavier_uniform',
            bias_initializer:str='zeros',
            reg_vals:dict=None,
            **kwargs,
            ):

        assert input_dims is not None, "NDNLayer: Must specify input_dims"
        assert num_filters is not None, "NDNLayer: Must specify num_filters"

        if len(kwargs) > 0:
            print("NDNLayer: unknown kwargs:", kwargs)

        super(NDNLayer, self).__init__()
        self.input_dims = input_dims
        self.num_filters = num_filters
        if filter_dims is None:
            self.filter_dims = deepcopy(input_dims)
        else:
            self.filter_dims = filter_dims
        
        output_dims = [num_filters, 1, 1, 1]
        self.output_dims = output_dims
        
        self.norm_type = norm_type
        self.pos_constraint = pos_constraint
        self.conv = False
        
        # Was this implemented correctly? Where should NLtypes (the dictionary) live?
        if NLtype in NLtypes:
            self.NL = NLtypes[NLtype]
        else:
            print("Nonlinearity undefined.")
            # return error

        self.shape = tuple([np.prod(self.filter_dims), self.num_filters])

        # Make layer variables
        self.weight = Parameter(torch.Tensor(size=self.shape))
        if bias:
            self.bias = Parameter(torch.Tensor(self.num_filters))
        else:
            self.register_buffer('bias', torch.zeros(self.num_filters))

        if self.pos_constraint:
            self.register_buffer("minval", torch.tensor(0.0))
            # Does this apply to both weights and biases? Should be separate?
            # How does this compare without explicit constraint? Maybe better, maybe worse...

        self.reg = Regularization( filter_dims=self.filter_dims, vals=reg_vals)

        if num_inh == 0:
            self.ei_mask = None
        else:
            self.register_buffer('ei_mask', torch.ones(self.num_filters))  
            self.ei_mask[-(num_inh+1):0] = -1

        self.reset_parameters( weights_initializer, bias_initializer )

    def reset_parameters(self, weights_initializer=None, bias_initializer=None, param=None) -> None:
        '''
        Initialize the weights and bias parameters of the layer.
        Args:
            weights_initializer: str, type of weight initialization to use
                options: 'xavier_uniform', 'xavier_normal', 'kaiming_uniform', 'kaiming_normal', 'orthogonal', 'zeros', 'ones'
            bias_initializer: str, type of bias initialization to use
        '''
        # Default initializer, although others possible
        if weights_initializer is None:
            weights_initializer = 'uniform'
        if bias_initializer is None:
            bias_initializer = 'zeros'

        if weights_initializer == 'uniform':
            if param is None:
                param = 5**0.5
            init.kaiming_uniform_(self.weight, a=param)
        elif weights_initializer == 'zeros':
            init.zeros_(self.weight)
        elif weights_initializer == 'normal':
            if param is None:
                mean = 0.0
                std = 1.0
            elif isinstance(param, list):
                mean, std = param
            else:
                RuntimeError("if using 'normal', param must be None or a list of length 2")
            init.normal_(self.weight, mean=mean, std=std)

        elif weights_initializer == 'xavier_uniform':  # also known as Glorot initialization
            if param is None:
                param = 1.0
            init.xavier_uniform_(self.weight, gain=param)  # set gain based on weights?
        elif weights_initializer == 'xavier_normal':  # also known as Glorot initialization
            if param is None:
                param = 1.0
            init.xavier_normal_(self.weight, gain=param)  # set gain based on weights?
        else:
            print('weights initializer not defined')

        if bias_initializer == 'zeros':
            init.zeros_(self.bias)
        elif bias_initializer == 'uniform':
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / np.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)          
        else:
            print('bias initializer not defined')
    
    def preprocess_weights(self):
        # Apply positive constraints
        if self.pos_constraint:
            w = torch.maximum(self.weight, self.minval)
        else:
            w = self.weight
        # Add normalization
        if self.norm_type > 0: # so far just standard filter-specific normalization
            w = F.normalize( w, dim=0 )
        return w

    def forward(self, x):
        # Pull weights and process given pos_constrain and normalization conditions
        w = self.preprocess_weights()
        # Simple linear processing and bias
        x = torch.matmul(x, w) + self.bias

        # Nonlinearity
        if self.NL is not None:
            x = self.NL(x)

        # Constrain output to be signed
        if self.ei_mask is not None:
            x = x * self.ei_mask

        return x 
        
    def compute_reg_loss(self):
        return self.reg.compute_reg_loss(self.preprocess_weights())

    def get_weights(self, to_reshape=True, time_reverse=False):
        ws = self.preprocess_weights().detach().cpu().numpy()
        num_filts = ws.shape[-1]
        if time_reverse:
            ws_tmp = np.reshape(ws, self.filter_dims + [num_filts])
            num_lags = self.filter_dims[3]
            if num_lags > 1:
                ws_tmp = ws_tmp[:, :, :, range(num_lags-1, -1, -1), :] # time reversal here
            ws = ws_tmp.reshape((-1, num_filts))
        if to_reshape:
            return ws.reshape(self.filter_dims + [num_filts]).squeeze()
        else:
            return ws.squeeze()

    def list_parameters(self):
        for nm, pp in self.named_parameters(recurse=False):
            if pp.requires_grad:
                print("      %s:"%nm, pp.size())
            else:
                print("      NOT FIT: %s:"%nm, pp.size())

    def set_parameters(self, name=None, val=None ):
        """
        Turn fitting for named params on or off.
        If name is none, do for whole layer.
        """
        assert isinstance(val, bool), 'val must be a boolean (True or False).'
        for nm, pp in self.named_parameters(recurse=False):
            if name is None:
                pp.requires_grad = val
            elif nm == name:
                pp.requires_grad = val

    def plot_filters( self, cmaps='gray', num_cols=8, row_height=2, time_reverse=True):
        """
        Plot the filters in the layer.
        Args:
            cmaps: str or colormap, colormap to use for plotting
            num_cols: int, number of columns to use in plot
            row_height: int, number of rows to use in plot
            time_reverse: bool, whether to reverse the time dimension
        """
        ws = self.get_weights(time_reverse=time_reverse)
        # check if 1d: otherwise return error for now
        num_filters = ws.shape[-1]
        if num_filters < 8:
            num_cols = num_filters
        num_rows = np.ceil(num_filters/num_cols).astype(int)
        if self.input_dims[2] == 1:
            import matplotlib.pyplot as plt
            # then 1-d spatiotemporal plots
            fig, ax = plt.subplots(nrows=num_rows, ncols=num_cols)
            fig.set_size_inches(16, row_height*num_rows)
            plt.tight_layout()
            for cc in range(num_filters):
                ax = plt.subplot(num_rows, num_cols, cc+1)
                plt.imshow(ws[:,:,cc].T, cmap=cmaps)
                ax.set_yticklabels([])
                ax.set_xticklabels([])
            plt.show()
        else:
            print("Not implemented plotting 3-d filters yet.")

class ConvLayer(NDNLayer):
    """
    Convolutional NDN Layer

    Args (required):
        input_dims: tuple or list of ints, (num_channels, height, width, lags)
        num_filters: number of output filters
        conv_width: width of convolutional kernel (int or list of ints)
    Args (optional):
        weight_init: str, 'uniform', 'normal', 'xavier', 'zeros', or None
        bias_init: str, 'uniform', 'normal', 'xavier', 'zeros', or None
        bias: bool, whether to include bias term
        NLtype: str, 'lin', 'relu', 'tanh', 'sigmoid', 'elu', 'none'

    """
    def __init__(self,
            input_dims=None,
            num_filters=None,
            conv_dims=None,
            filter_dims=None,
            stride=None,
            dilation=1,
            **kwargs,
            ):

        assert input_dims is not None, "ConvLayer: Must specify input_dims"
        assert num_filters is not None, "ConvLayer: Must specify num_filters"
        assert (conv_dims is not None) or (filter_dims is not None), "ConvLayer: conv_dims or filter_dims must be specified"
        
        if conv_dims is None:
            conv_dims = filter_dims[1:]

        if filter_dims is None:
            if isinstance(conv_dims, int):
                filter_dims = [input_dims[0], conv_dims, conv_dims, input_dims[3]]
            elif isinstance(conv_dims, list):
                if len(conv_dims) == 1:
                    filter_dims = [input_dims[0], conv_dims[0], conv_dims[0], input_dims[3]]
                elif len(conv_dims) == 2:
                    filter_dims = [input_dims[0], conv_dims[0], conv_dims[1], input_dims[3]]
                elif len(conv_dims) == 3:
                    filter_dims = [input_dims[0], conv_dims[0], conv_dims[1], conv_dims[2]]
                else:
                    raise ValueError('conv_width must be int or list of length 1, 2, or 3.')

        super(ConvLayer, self).__init__(input_dims,
            num_filters,
            filter_dims=filter_dims,
            **kwargs,
            )

        output_dims = [num_filters, 1, 1, 1]
        output_dims[1:3] = input_dims[1:3]
        self.output_dims = output_dims

        if input_dims[2] == 1:  # then 1-d spatial
            filter_dims[2] = 1
            self.is1D = (self.input_dims[2] == 1)

        self.filter_dims = filter_dims
        # Checks to ensure cuda-bug with large convolutional filters is not activated #1
        assert self.filter_dims[1] < self.input_dims[1], "Filter widths must be smaller than input dims."
        # Check if 1 or 2-d convolution required
        if self.input_dims[2] > 1:
            assert self.filter_dims[2] < self.input_dims[2], "Filter widths must be smaller than input dims."

        if stride is None:
            self.stride = 1   
        else: 
            self.stride = stride

        if dilation is None:
            self.dilation = 1
        else:
            self.dilation = dilation

        if self.stride > 1:
            print('Warning: Manual padding not yet implemented when stride > 1')
            self.padding = 0
        else:
            w = self.filter_dims[1:3] # handle 2D if necessary
            self.padding = (w[0]//2, (w[0]-1+w[0]%2)//2, w[1]//2, (w[1]-1+w[1]%2)//2)

        # Combine filter and temporal dimensions for conv -- collapses over both
        self.folded_dims = self.input_dims[0]*self.input_dims[3]
        self.num_outputs = np.prod(self.output_dims)

    def forward(self, x):
        # Reshape weight matrix and inputs (note uses 4-d rep of tensor before combinine dims 0,3)
        s = x.reshape([-1]+self.input_dims).permute(0,1,4,2,3)
        w = self.preprocess_weights().reshape(self.filter_dims+[-1]).permute(4,0,3,1,2)
        # puts output_dims first, as required for conv

        # Collapse over irrelevant dims for dim-specific convs
        if self.is1D:
            s = torch.reshape( s, (-1, self.folded_dims, self.input_dims[1]) )
            y = F.conv1d(
                F.pad(s, self.padding, "constant", 0), # we do our own padding
                torch.reshape( w, (-1, self.folded_dims, self.filter_dims[1]) ), 
                bias=self.bias,
                stride=self.stride, dilation=self.dilation)
        else:
            s = torch.reshape( s, (-1, self.folded_dims, self.input_dims[1], self.input_dims[2]) )
            y = F.conv2d(
                F.pad(s, self.padding, "constant", 0), # we do our own padding
                torch.reshape( w, (-1, self.folded_dims, self.filter_dims[1], self.filter_dims[2]) ), 
                bias=self.bias,
                stride=self.stride, padding=self.padding, dilation=self.dilation)

        y = torch.reshape(y, (-1, self.num_outputs))
        
        # Nonlinearity
        if self.NL is not None:
            y = self.NL(y)
        
        if self.ei_mask is not None:
            y = y * self.ei_mask

        return y



class DivNormLayer(NDNLayer):
    """
    Divisive normalization implementation: not explicitly convolutional
    

    """ 

    def __init__(self,
            input_dims=None,
            num_filters=None,
            filter_dims=None,
            pos_constraint=True,
            **kwargs,
        ):

        assert (input_dims is not None) or (num_filters is not None), "DivNormLayer: Must specify either input_dims or num_filters"
        
        if not pos_constraint:
            pos_constraint = True

        if num_filters is None:            
            num_filters = input_dims[0]

        if input_dims is None:
            input_dims = [num_filters, 1, 1, 1] # assume non-convolutional

        # number of filters (and size of filters) is set by channel dims on the input
        
        filter_dims = [num_filters, 1, 1, 1]

        super(DivNormLayer, self).__init__(input_dims,
            num_filters,
            filter_dims=filter_dims,
            pos_constraint=pos_constraint, **kwargs)

        self.output_dims = self.input_dims
        self.num_outputs = np.prod(self.output_dims)

        self.weight.data.fill_(1/self.num_outputs)
        self.bias.data.fill_(0.5)

    def forward(self, x):
        # TODO: make if statement for 1-d or 2-d, 3-d indpendent of x, should be specified by a property

        # Pull weights and process given pos_constrain and normalization conditions
        w = self.preprocess_weights()

        # Nonlinearity (apply first)
        if self.NL is not None:
            x = self.NL(x)

        x = x.reshape([-1] + self.input_dims)

        # Linear processing to create divisive drive
        xdiv = torch.einsum('nc...,ck->nk...', x, w)

        if len(x.shape)==2:
            xdiv = xdiv + self.bias[None,:]
        elif len(x.shape)==3:
            xdiv = xdiv + self.bias[None,:,None] # is 1D convolutional
        elif len(x.shape)==4:
            xdiv = xdiv + self.bias[None,:,None,None] # is 2D convolutional
        elif len(x.shape)==5:
            xdiv = xdiv + self.bias[None,:,None,None,None] # is 2D convolutional
        else:
            raise NotImplementedError('DivNormLayer only supports 2D, 3D, and 4D tensors')
            
        # apply divisive drive
        x = x / xdiv.clamp_(0.001) # divide
        
        x = x.reshape((-1, self.num_outputs))
        return x


class ReadoutLayer(NDNLayer):

    def __init__(self, 
            input_dims=None,
            num_filters=None,
            filter_dims=None, 
            batch_sample=True,
            init_mu_range=0.1,
            init_sigma=0.1,
            gauss_type: str='uncorrelated', # 'isotropic', 'uncorrelated', or 'full'
            align_corners=False,
            **kwargs):

        assert input_dims is not None, "ReadoutLayer: Must specify input_dims"
        assert num_filters is not None, "ReadoutLayer: Must specify num_filters"

        # Make sure filter_dims is filled in, and to single the spatial filter dimensions
        if filter_dims is None:
            filter_dims = deepcopy(input_dims)
        
        filter_dims[1:3] = [1,1]
        print("filter dims:", filter_dims)
        assert filter_dims[3] == 1, 'Cant handle temporal filter dims here, yet.'
        print('kwarg keys:', kwargs.keys())
        super(ReadoutLayer,self).__init__(input_dims,
            num_filters,
            filter_dims=filter_dims,
            **kwargs)
        
        # Determine whether one- or two-dimensional readout
        if input_dims[2] == 1:
            self.num_space_dims = 1
        else:
            self.num_space_dims = 2

        # Additional readout parameters (below)
        if self.pos_constraint:
            self.register_buffer("minval", torch.tensor(0.0))
            # Does this apply to both weights and biases? Should be separate?

        # self.flatten = nn.Flatten()
        self.batch_sample = batch_sample
            
        self.init_mu_range = init_mu_range
        self.init_sigma = init_sigma

        if self.init_mu_range > 1.0 or self.init_mu_range <= 0.0 or self.init_sigma <= 0.0:
            raise ValueError("either init_mu_range doesn't belong to [0.0, 1.0] or init_sigma_range is non-positive")
        
        if self.num_space_dims == 1:
            self.gauss_type = 'isotropic'  # only 1-d to sample sigma in
        else:
            self.gauss_type = gauss_type
        self.align_corners = align_corners

        # position grid shape
        self.grid_shape = (1, self.num_filters, 1, self.num_space_dims)
        if self.gauss_type == 'full':
            self.sigma_shape = (1, self.num_filters, 2, 2)
        elif self.gauss_type == 'uncorrelated':
            self.sigma_shape = (1, self.num_filters, 1, 2)
        elif self.gauss_type == 'isotropic':
            self.sigma_shape = (1, self.num_filters, 1, 1)
        else:
            raise ValueError(f'gauss_type "{self.gauss_type}" not known')

        # initialize means and spreads
        self._mu = Parameter(torch.Tensor(*self.grid_shape))  # mean location of gaussian for each neuron
        self.sigma = Parameter(torch.Tensor(*self.sigma_shape))  # standard deviation for gaussian for each neuron

        self.initialize_spatial_mapping()

    @property
    def features(self):

        if self.pos_constraint:
            feat = F.relu(self.weight)
        else:
            feat = self.weight
        
        return feat

    @property
    def grid(self):
        return self.sample_grid(batch_size=1, sample=False)

    @property
    def mu(self):
        return self._mu

    def initialize_spatial_mapping(self):
        """
        Initializes the mean, and sigma of the Gaussian readout 
        """
        self._mu.data.uniform_(-self.init_mu_range, self.init_mu_range)  # random initializations uniformly spread....
        if self.gauss_type != 'full':
            self.sigma.data.fill_(self.init_sigma)
        else:
            self.sigma.data.uniform_(-self.init_sigma, self.init_sigma)
        
        #self.weight.data.fill_(1 / self.input_dims[0])

        if self.bias is not None:
            self.bias.data.fill_(0)

    def sample_grid(self, batch_size, sample=None):
        """
        Returns the grid locations from the core by sampling from a Gaussian distribution
        DAN: more specifically, it returns sampled positions for each batch over all elements, given mus and sigmas
        DAN: if not 'sample', it just gives mu back (so testing has no randomness)
        DAN: this is some funny bit of code, but don't think I need to touch it past what I did
        Args:
            batch_size (int): size of the batch
            sample (bool/None): sample determines whether we draw a sample from Gaussian distribution, N(mu,sigma), defined per neuron
                            or use the mean, mu, of the Gaussian distribution without sampling.
                           if sample is None (default), samples from the N(mu,sigma) during training phase and
                             fixes to the mean, mu, during evaluation phase.
                           if sample is True/False, overrides the model_state (i.e training or eval) and does as instructed
        """
        with torch.no_grad():
            self.mu.clamp(min=-1, max=1)  # at eval time, only self.mu is used so it must belong to [-1,1]
            if self.gauss_type != 'full':
                self.sigma.clamp(min=0)  # sigma/variance is always a positive quantity

        grid_shape = (batch_size,) + self.grid_shape[1:]
                
        sample = self.training if sample is None else sample
        if sample:
            norm = self.mu.new(*grid_shape).normal_()
        else:
            norm = self.mu.new(*grid_shape).zero_()  # for consistency and CUDA capability

        if self.num_space_dims == 1:
            # this is dan's 1-d kluge code -- necessary because grid_sampler has to have last dimension of 2. maybe problem....
            grid2d = norm.new_zeros(*(grid_shape[:3]+(2,)))  # for consistency and CUDA capability
            grid2d[:,:,:,0] = (norm * self.sigma + self.mu).clamp(-1,1).squeeze(-1)
            return grid2d
            #return (norm * self.sigma + self.mu).clamp(-1,1) # this needs second dimension
        else:
            if self.gauss_type != 'full':
                return (norm * self.sigma + self.mu).clamp(-1,1) # grid locations in feature space sampled randomly around the mean self.mu
            else:
                return (torch.einsum('ancd,bnid->bnic', self.sigma, norm) + self.mu).clamp_(-1,1) # grid locations in feature space sampled randomly around the mean self.mu

    def passive_readout(self):
        """This will not fit mu and std, but set them to zero. It will pass identities in,
        so number of input filters must equal number of readout units"""

        assert self.filter_dims[0] == self.output_dims[0], "Must have #filters = #output units."
        # Set parameters for default readout
        self.sigma.data.fill_(0)
        self._mu.data.fill_(0)
        self.weight.data.fill_(0)
        for nn in range(self.num_filters):
            self.weight.data[nn,nn] = 1

        self.set_parameters(val=False)
        

    def forward(self, x, shift=None):
        """
        Propagates the input forwards through the readout
        Args:
            x: input data
            sample (bool/None): sample determines whether we draw a sample from Gaussian distribution, N(mu,sigma), defined per neuron
                            or use the mean, mu, of the Gaussian distribution without sampling.
                           if sample is None (default), samples from the N(mu,sigma) during training phase and
                             fixes to the mean, mu, during evaluation phase.
                           if sample is True/False, overrides the model_state (i.e training or eval) and does as instructed
            shift (bool): shifts the location of the grid (from eye-tracking data)
            out_idx (bool): index of neurons to be predicted

        Returns:
            y: neuronal activity
        """
        x = x.reshape([-1]+self.input_dims[:3])

        N, c, w, h = x.size()   # N is number of time points -- note that its full dimensional....
        c_in, w_in, h_in = self.input_dims[:3]
        if (c_in, w_in, h_in) != (c, w, h):
            raise ValueError("the specified feature map dimension is not the readout's expected input dimension")
        
        feat = self.features  # this is the filter weights for each unit
        feat = feat.reshape(1, c, self.num_filters)
        bias = self.bias
        outdims = self.num_filters

        if self.batch_sample:
            # sample the grid_locations separately per sample per batch
            grid = self.sample_grid(batch_size=N, sample=self.training)  # sample determines sampling from Gaussian
        else:
            # use one sampled grid_locations for all sample in the batch
            grid = self.sample_grid(batch_size=1, sample=self.training).expand(N, outdims, 1, self.num_space.dims)
        
        if shift is not None:
            # shifter is run outside the readout forward
            grid = grid + shift[:, None, None, :]

        y = F.grid_sample(x, grid, align_corners=self.align_corners)

        y = (y.squeeze(-1) * feat).sum(1).view(N, outdims)

        if self.bias is not None:
            y = y + bias
        
        if self.NL is not None:
            y = self.NL(y)
        
        return y

    def set_readout_locations(self, locs):
        """This hasn't been tested yet, but should be self-explanatory"""
        if len(locs.shape) == 1:
            locs = np.expand_dims(locs,1)
        NC, num_dim = locs.shape
        assert num_dim == self.num_space_dims, 'Incorrect number of spatial dimensions'
        assert NC == self.num_filters, 'Incorrect number of neuron positions.'
        self._mu = deepcopy(locs)

    def get_readout_locations(self):
        """Currently returns center location and sigmas, as list"""
        return self.mu.detach().cpu().numpy().squeeze(), self.sigma.detach().cpu().numpy().squeeze()    

class STconvLayer(ConvLayer):
    """
    Spatio-temporal convolutional layer.
    STConv Layers overload the batch dimension and assume they are contiguous in time.

    Args:
        input_dims (list of ints): input dimensions [C, W, H, T]
            This is, of course, not the real input dimensions, because the batch dimension is assumed to be contiguous.
            The real input dimensions are [B, C, W, H, 1], T specifies the number of lags
        num_filters (int): number of filters
        filter_dims (list of ints): filter dimensions [C, w, h, T]
            w < W, h < H
        stride (int): stride of convolution

    
    """ 

    def __init__(self,
        input_dims=None, # [C, W, H, T]
        num_filters=None, # int
        conv_dims=None, # [w,h,t]
        filter_dims=None, # [C, w, h, t]
        temporal_tent_spacing=None,
        stride=1,
        dilation=1,
        **kwargs):
        
        assert input_dims is not None, "STConvLayer: input_dims must be specified"
        assert num_filters is not None, "STConvLayer: num_filters must be specified"
        assert (conv_dims is not None) or (filter_dims is not None), "STConvLayer: conv_dims or filter_dims must be specified"
        
        if conv_dims is None:
            conv_dims = filter_dims[1:]

        # If tent-basis, figure out how many lag-dimensions using tent_basis transform
        tent_basis = None
        if temporal_tent_spacing is not None:
            from utils import tent_basis_generate
            num_lags = conv_dims[2]
            tent_basis = tent_basis_generate(np.arange(0, num_lags, temporal_tent_spacing))
            num_lag_params = tent_basis.shape[1]
            print('STconv: num_lag_params =', num_lag_params)
            conv_dims[2] = num_lag_params

        if filter_dims is None:
            filter_dims = [input_dims[0]] + conv_dims
        else:            
            filter_dims[1:] = conv_dims

        assert stride == 1, 'Cannot handle greater strides than 1.'
        assert dilation == 1, 'Cannot handle greater dilations than 1.'

        # All parameters of filter (weights) should be correctly fit in layer_params
        super(STconvLayer, self).__init__(input_dims,
            num_filters, conv_dims, stride=stride, dilation=dilation, **kwargs)

        self.num_lags = self.input_dims[3]
        self.input_dims[3] = 1  # take lag info and use for temporal convolution

        if tent_basis is not None:
            self.register_buffer('tent_basis', torch.Tensor(tent_basis.T))
        else:
            self.tent_basis = None

        # Check if 1 or 2-d convolution required
        self.is1D = (self.input_dims[2] == 1)
        # "1D" really means a 2D convolution (1D space, 1D time) since time is handled with
        # convolutions instead of embedding lags

        # Do spatial padding by hand -- will want to generalize this for two-ds
        if self.is1D:
            self.padding = (self.filter_dims[1]//2, (self.filter_dims[1] - 1 + self.filter_dims[1]%2)//2,
                self.num_lags-1, 0)
        else:
            # Checks to ensure cuda-bug with large convolutional filters is not activated #2
            assert self.filter_dims[2] < self.input_dims[2], "Filter widths must be smaller than input dims."

            self.padding = (self.filter_dims[1]//2, (self.filter_dims[1] - 1 + self.filter_dims[1]%2)//2,
                self.filter_dims[2]//2, (self.filter_dims[2] - 1 + self.filter_dims[2]%2)//2,
                self.num_lags-1, 0)


        # Combine filter and temporal dimensions for conv -- collapses over both
        self.num_outputs = np.prod(self.output_dims)

    #     # Initialize weights
    #     self.initialize_weights()

    #     # NOTE FROM DAN: this can inherit the NDNlayer initialization (specificied in layer_dicts) rather than automatically do thi
    #     # just replace this initialize_weights with reset_parameters: no need to overload: will use the one in NDNlayer
    # # END STconvLayer.__init__

    # def initialize_weights(self):
    #     """Initialize weights and biases"""
    #     nn.init.xavier_normal_(self.weight.data)
    #     if self.bias is not None:
    #         self.bias.data.fill_(0)

    def forward(self, x):
        # Reshape stim matrix LACKING temporal dimension [bcwh] 
        # and inputs (note uses 4-d rep of tensor before combinine dims 0,3)
        # pytorch likes 3D convolutions to be [B,C,T,W,H].
        # I benchmarked this and it'sd a 20% speedup to put the "Time" dimension first.

        w = self.preprocess_weights()
        if self.is1D:
            s = x.reshape([-1] + self.input_dims[:3]).permute(3,1,0,2) # [B,C,W,1]->[1,C,B,W]
            w = w.reshape(self.filter_dims[:2] + [self.filter_dims[3]] +[-1]).permute(3,0,2,1) # [C,H,T,N]->[N,C,T,W]
            # time-expand using tent-basis if it exists
            if self.tent_basis is not None:
                w = torch.einsum('nctw,tz->nczw', w, self.tent_basis)

            y = F.conv2d(
                F.pad(s, self.padding, "constant", 0),
                w, 
                bias=self.bias,
                stride=self.stride, dilation=self.dilation)
            y = y.permute(2,1,3,0) # [1,N,B,W] -> [B,N,W,1]

        else:
            s = x.reshape([-1] + self.input_dims).permute(4,1,0,2,3) # [1,C,B,W,H]
            w = w.reshape(self.filter_dims + [-1]).permute(4,0,3,1,2) # [N,C,T,W,H]
            
            # time-expand using tent-basis if it exists
            if self.tent_basis is not None:
                w = torch.einsum('nctwh,tz->nczwh', w, self.tent_basis)

            y = F.conv3d(
                F.pad(s, self.padding, "constant", 0),
                w, 
                bias=self.bias,
                stride=self.stride, dilation=self.dilation)
            y = y.permute(2,1,3,4,0)    
        
        #y = torch.reshape(y, (-1, self.num_outputs))
        y = y.reshape((-1, self.num_outputs))

        # Nonlinearity
        if self.NL is not None:
            y = self.NL(y)
        return y

    def plot_filters( self, cmaps='gray', num_cols=8, row_height=2, time_reverse=False):
        # Overload plot_filters to automatically time_reverse
        super(STconvLayer, self).plot_filters( 
            cmaps=cmaps, num_cols=num_cols, row_height=row_height, 
            time_reverse=time_reverse)


# class FixationLayer(NDNLayer):
#     """
#     FixationLayer for fixation 
#     """
#     def __init__(self, layer_params, reg_vals=None):
#         """layer weights become the shift for each fixation, sigma is constant over each dimension"""
#         #self.num_fixations = layer_params['input_dims'][0]
#         assert np.prod(layer_params['input_dims'][1:]) == 1, 'something wrong with fix-layer input_dims'
#         layer_params['filter_dims'] = deepcopy(layer_params['input_dims'])
#         assert layer_params['num_filters'] in [1,2], 'incorrect num_filters in fix-layer'
#         layer_params['bias'] = False
#         layer_params['NLtype'] = 'lin'  # dummy variable: not usred

#         super(FixationLayer,self).__init__(layer_params)
#         assert layer_params['filter_dims'][3] == 1, 'Cant handle temporal filter dims here, yet.'

#         # Determine whether one- or two-dimensional readout
#         self.num_space_dims = layer_params['num_filters']
        
#         # self.flatten = nn.Flatten()
#         if 'batch_sample' in layer_params.keys():
#             self.batch_sample = layer_params['batch_sample']
#         else:
#             self.batch_sample = True
#         if 'init_sigma' in layer_params.keys(): 
#             self.init_sigma = layer_params['init_sigma']
#         else:
#             self.init_sigma = 0.5

#         self.single_sigma = False
#         if 'single_sigma' in layer_params.keys():
#             self.single_sigma = layer_params['single_sigma']
        
#         #self.sample = False  # starts without sampling be default
#         # shared sigmas across all fixations
#         if self.single_sigma:
#             self.sigmas = Parameter(torch.Tensor(self.num_space_dims)) 
#         else:
#             # individual sigmas for each fixation 
#             self.sigmas = Parameter(torch.Tensor(self.filter_dims[0],1))  
        
#         self.sigmas.data.fill_(self.init_sigma)

#         self.sample = False
#     # END FixationLayer.__init__

#     def forward(self, x, shift=None):
#         """
#         The input is the sampled fixation-stim
#             y: neuronal activity
#         """
#         # with torch.no_grad():
#         # self.weight.clamp(min=-1, max=1)  # at eval time, only self.mu is used so it must belong to [-1,1]
#         # self.sigmas.clamp(min=0)  # sigma/variance is always a positive quantity

#         N = x.shape[0]  # batch size
#         # If using X-matrix to extract relevant weights
#         # y = (x@self.weight) 
#         # use indexing: x is just a list of weight indices

#         #y = F.tanh(self.weight[x,:]) #* self.spatial_mults  # deprecated?
#         y = torch.tanh(self.weight[x,:]) 

#         if self.single_sigma:
#             s = self.sigmas**2 
#         else:
#             s = self.sigmas[x]**2

#         # this will be batch-size x num_spatial dims (corresponding to mean loc)        
#         if self.sample:  # can turn off sampling, even with training
#             if self.training:
#                 # add sigma-like noise around mean locations
#                 if self.batch_sample:
#                     sample_shape = (1,) + (self.num_space_dims,)
#                     gaus_sample = y.new(*sample_shape).normal_().repeat(N,1)
#                 else:
#                     sample_shape = (N,) + (self.num_space_dims,)
#                     gaus_sample = y.new(*sample_shape).normal_()
                
#                 y = (gaus_sample * s + y).clamp(-1,1)

#         return y
