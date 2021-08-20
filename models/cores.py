import warnings
import numpy as np
from collections import OrderedDict, Iterable

import torch
from torch import nn
from torch.nn import functional as F

from pytorch_lightning import LightningModule

# import regularizers
import neureye.models.regularizers as regularizers
from neureye.models.layers import AdaptiveELU, divNorm

""" 
Base classes:
"""
class Core(LightningModule):
    def initialize(self):
        raise NotImplementedError("Not initializing")

    def __repr__(self):
        s = super().__repr__()
        s += " [{} regularizers: ".format(self.__class__.__name__)
        ret = []
        for attr in filter(lambda x: "gamma" in x or "skip" in x, dir(self)):
            ret.append("{} = {}".format(attr, getattr(self, attr)))
        return s + "|".join(ret) + "]\n"
        
    def plot_filters(model, sort=False):
        import numpy as np
        import matplotlib.pyplot as plt  # plotting
        # ei_mask = model.features.layer0.eimask.ei_mask.detach().cpu().numpy()
        
        w = model.features.layer0.conv.weight.detach().cpu().numpy()
        ei_mask = np.ones(w.shape[0])
        sz = w.shape
        # w = model.features.weight.detach().cpu().numpy()
        w = w.reshape(sz[0], sz[1], sz[2]*sz[3])
        nfilt = w.shape[0]
        if type(sort) is np.ndarray:
            cinds = sort
        elif sort:
            n = np.asarray([w[i,:].abs().max().detach().numpy() for i in range(nfilt)])
            cinds = np.argsort(n)[::-1][-len(n):]
        else:
            cinds = np.arange(0, nfilt)

        sx = np.ceil(np.sqrt(nfilt*2))
        sy = np.round(np.sqrt(nfilt*2))
        # sx,sy = U.get_subplot_dims(nfilt*2)
        mod2 = sy % 2
        sy += mod2
        sx -= mod2

        plt.figure(figsize=(10,10))
        for cc,jj in zip(cinds, range(nfilt)):
            plt.subplot(sx,sy,jj*2+1)
            wtmp = np.squeeze(w[cc,:])
            bestlag = np.argmax(np.std(wtmp, axis=1))
            plt.imshow(np.reshape(wtmp[bestlag,:], (sz[2], sz[3])), interpolation=None, )
            wmax = np.argmax(wtmp[bestlag,:])
            wmin = np.argmin(wtmp[bestlag,:])
            plt.axis("off")

            plt.subplot(sx,sy,jj*2+2)
            if ei_mask[cc]>0:
                plt.plot(wtmp[:,wmax], 'b-')
                plt.plot(wtmp[:,wmin], 'b--')
            else:
                plt.plot(wtmp[:,wmax], 'r-')
                plt.plot(wtmp[:,wmin], 'r--')

            plt.axhline(0, color='k')
            plt.axvline(bestlag, color=(.5, .5, .5))
            plt.axis("off")


class Core2d(Core):
    """
    2D convolutional core base class
    """
    def initialize(self, cuda=False):
        self.apply(self.init_conv)

    @staticmethod
    def init_conv(m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0)


"""
STACKED 2D Convolutional Core (based on Sinz lab code)
"""
class Stacked2dCore(Core2d):
    def __init__(
        self,
        input_channels,
        hidden_channels,
        input_kern,
        hidden_kern,
        layers=1,
        gamma_hidden=0,
        gamma_input=0.0,
        gamma_center=.1, # regularize the first conv layer to be centered
        skip=0,
        input_regularizer="RegMats",
        hidden_regularizer="RegMats",
        input_reg_types=["d2xt", "local","center"],
        input_reg_amt=[.25,.25,.5],
        hidden_reg_types=["d2x", "local", "center"],
        hidden_reg_amt=[.33,.33,.33],
        stack=None,
        use_avg_reg=True,
    ):
        """
        Args:
            input_channels:     Integer, number of input channels as in
            hidden_channels:    Number of hidden channels (i.e feature maps) in each hidden layer
            input_kern:     kernel size of the first layer (i.e. the input layer)
            hidden_kern:    kernel size of each hidden layer's kernel
            layers:         number of layers
            gamma_hidden:   regularizer factor for group sparsity
            gamma_input:    regularizer factor for the input weights (default: LaplaceL2, see neuralpredictors.regularizers)
            skip:           Adds a skip connection
            stack:        Int or iterable. Selects which layers of the core should be stacked for the readout.
                            default value will stack all layers on top of each other.
                            Implemented as layers_to_stack = layers[stack:]. thus:
                                stack = -1 will only select the last layer as the readout layer.
                                stack of -2 will read out from the last two layers.
                                And stack of 1 will read out from layer 1 (0 indexed) until the last layer.

            use_avg_reg:    bool. Whether to use the averaged value of regularizer(s) or the summed.
        """

        super().__init__()

        # regularizer_config = (
        #     dict(padding=laplace_padding, kernel=input_kern)
        #     if input_regularizer == "GaussianLaplaceL2"
        #     else dict(padding=laplace_padding)
        # )
        # self._input_weights_regularizer = regularizers.__dict__[input_regularizer](**regularizer_config)
        regularizer_config = {'dims': [input_channels, input_kern,input_kern],
                            'type': input_reg_types, 'amount': input_reg_amt}
        self._input_weights_regularizer = regularizers.__dict__["RegMats"](**regularizer_config)

        regularizer_config = {'dims': [hidden_channels, hidden_kern,hidden_kern],
                            'type': hidden_reg_types, 'amount': hidden_reg_amt}
        self._hidden_weights_regularizer = regularizers.__dict__["RegMats"](**regularizer_config)

        self.layers = layers
        self.gamma_input = gamma_input
        self.gamma_hidden = gamma_hidden
        self.gamma_center = gamma_center
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.skip = skip
        self.use_avg_reg = use_avg_reg

        if stack is None:
            self.stack = range(self.layers)
        else:
            self.stack = [*range(self.layers)[stack:]] if isinstance(stack, int) else stack

        # # center regularization
        # regw = 1 - regularizers.gaussian2d(input_kern,sigma=input_kern//2)
        # self.register_buffer("center_reg_weights", torch.tensor(regw))

    def forward(self, input_):
        ret = []
        for l, feat in enumerate(self.features):
            do_skip = l >= 1 and self.skip > 1
            input_ = feat(input_ if not do_skip else torch.cat(ret[-min(self.skip, l) :], dim=1))
            ret.append(input_)

        return torch.cat([ret[ind] for ind in self.stack], dim=1)

    def input_reg(self):
        return self._input_weights_regularizer(self.features[0].conv.weight)

    def hidden_reg(self):
        ret = 0
        for l in range(1, self.layers):
            ret = ret + self._hidden_weights_regularizer(self.features[l].conv.weight)
        return self.gamma_hidden * self.group_sparsity() + ret
    # def center_reg(self):
    #     ret = torch.einsum('ijxy,xy->ij', self.features[0].conv.weight, self.center_reg_weights).pow(2).mean()
    #     return ret

    def group_sparsity(self):
        ret = 0
        for l in range(1, self.layers):
            ret = ret + self.features[l].conv.weight.pow(2).sum(3, keepdim=True).sum(2, keepdim=True).sqrt().mean()
        return ret / ((self.layers - 1) if self.layers > 1 else 1)

    def regularizer(self):
        return self.hidden_reg() + self.gamma_input * self.input_reg()

    @property
    def outchannels(self):
        if self.stack:
            ret = len(self.stack) * self.hidden_channels
        else:
            ret = len(self.features) * self.hidden_channels
        return ret
    
    def plot_filters(model, sort=False):
        import numpy as np
        import matplotlib.pyplot as plt  # plotting
        # ei_mask = model.features.layer0.eimask.ei_mask.detach().cpu().numpy()
        
        w = model.features.layer0.conv.weight.detach().cpu().numpy()
        ei_mask = np.ones(w.shape[0])
        sz = w.shape
        # w = model.features.weight.detach().cpu().numpy()
        w = w.reshape(sz[0], sz[1], sz[2]*sz[3])
        nfilt = w.shape[0]
        if type(sort) is np.ndarray:
            cinds = sort
        elif sort:
            n = np.asarray([w[i,:].abs().max().detach().numpy() for i in range(nfilt)])
            cinds = np.argsort(n)[::-1][-len(n):]
        else:
            cinds = np.arange(0, nfilt)

        sx = np.ceil(np.sqrt(nfilt*2))
        sy = np.round(np.sqrt(nfilt*2))
        # sx,sy = U.get_subplot_dims(nfilt*2)
        mod2 = sy % 2
        sy += mod2
        sx -= mod2

        plt.figure(figsize=(10,10))
        for cc,jj in zip(cinds, range(nfilt)):
            plt.subplot(sx,sy,jj*2+1)
            wtmp = np.squeeze(w[cc,:])
            bestlag = np.argmax(np.std(wtmp, axis=1))
            plt.imshow(np.reshape(wtmp[bestlag,:], (sz[2], sz[3])), interpolation=None, )
            wmax = np.argmax(wtmp[bestlag,:])
            wmin = np.argmin(wtmp[bestlag,:])
            plt.axis("off")

            plt.subplot(sx,sy,jj*2+2)
            if ei_mask[cc]>0:
                plt.plot(wtmp[:,wmax], 'b-')
                plt.plot(wtmp[:,wmin], 'b--')
            else:
                plt.plot(wtmp[:,wmax], 'r-')
                plt.plot(wtmp[:,wmin], 'r--')

            plt.axhline(0, color='k')
            plt.axvline(bestlag, color=(.5, .5, .5))
            plt.axis("off")



class Stacked2dDivNorm(Stacked2dCore):
    def __init__(
        self,
        input_channels=10,
        hidden_channels=10,
        input_kern=9,
        hidden_kern=9,
        activation="elu",
        final_nonlinearity=True,
        bias=False,
        pad_input=True,
        hidden_padding=None,
        group_norm=True,
        num_groups=2,
        weight_norm=True,
        hidden_dilation=1,
        **kwargs):

        self.save_hyperparameters()

        super().__init__(input_channels,hidden_channels,input_kern,hidden_kern,**kwargs)

        self.features = nn.Sequential()
        
        if activation=="elu":
            self.activation = AdaptiveELU(0.0,1.0)
        elif activation=="relu":
            self.activation = nn.ReLU()

        # --- first layer
        layer = OrderedDict()
        if weight_norm:
            layer["conv"] = nn.utils.weight_norm(nn.Conv2d(
                input_channels,
                hidden_channels,
                input_kern,
                padding=input_kern // 2 if pad_input else 0,
                bias=bias and not group_norm),
                dim=0, name='weight')
        else:
            layer["conv"] = nn.Conv2d(
                input_channels,
                hidden_channels,
                input_kern,
                padding=input_kern // 2 if pad_input else 0,
                bias=bias and not group_norm,
            )

        if self.layers > 1 or final_nonlinearity:
            layer["nonlin"] = self.activation
            layer["norm"] = divNorm(hidden_channels)

        self.features.add_module("layer0", nn.Sequential(layer))

        # --- other layers
        if not isinstance(hidden_kern, Iterable):
            hidden_kern = [hidden_kern] * (self.layers - 1)

        for l in range(1, self.layers):
            layer = OrderedDict()

            hidden_padding = ((hidden_kern[l - 1] - 1) * hidden_dilation + 1) // 2
            if weight_norm:
                layer["conv"] = nn.utils.weight_norm(nn.Conv2d(
                    hidden_channels if not self.skip > 1 else min(self.skip, l) * hidden_channels,
                    hidden_channels,
                    hidden_kern[l - 1],
                    padding=hidden_padding,
                    bias=bias,
                    dilation=hidden_dilation),
                    dim=0)

            else:
                layer["conv"] = nn.Conv2d(
                    hidden_channels if not self.skip > 1 else min(self.skip, l) * hidden_channels,
                    hidden_channels,
                    hidden_kern[l - 1],
                    padding=hidden_padding,
                    bias=bias,
                    dilation=hidden_dilation,
                )

            if final_nonlinearity or l < self.layers - 1:
                layer["nonlin"] = self.activation #AdaptiveELU(elu_xshift, elu_yshift)

            if group_norm:
                layer["norm"] = nn.GroupNorm(num_groups, hidden_channels)

            self.features.add_module("layer{}".format(l), nn.Sequential(layer))

        # center regularization
        regw = 1 - regularizers.gaussian2d(input_kern,sigma=input_kern//4)
        self.register_buffer("center_reg_weights", torch.tensor(regw))

        self.apply(self.init_conv)