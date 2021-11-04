#%% 
import sys, os

# setup paths
iteration = 2 # which version of this tutorial to run (in case want results in different dirs)
NBname = 'example_2D_eyetracking{}'.format(iteration)

myhost = os.uname()[1] # get name of machine
print("Running on Computer: [%s]" %myhost)

if myhost=='mt': # this is sigur
    sys.path.insert(0, '/home/jake/Repos/')
    dirname = os.path.join('.', 'checkpoints')
    datadir = '/home/jake/Datasets/Mitchell/stim_movies'
elif myhost=='bancanus': # this is jake's workstation
    sys.path.insert(0, '/home/jake/Data/Repos/')
    datadir = '/home/jake/Data/Datasets/MitchellV1FreeViewing/stim_movies/'
    dirname = os.path.join('.', 'checkpoints')
else:
    sys.path.insert(0, '/home/dbutts/Code/') # you need this Repo, NDN3, and V1FreeViewingCode
    datadir = './MitchellLabFreeViewing'  # the datadir is part of the repository in this tutorial, but can be somewhere else
    # Working directory -- this determines where models and checkpoints are saved
    dirname = '/home/dbutts/V1/Monocular/'

import numpy as np
import torch

import matplotlib.pyplot as plt  # plotting

# the dataset
import datasets.mitchell.pixel as datasets

# shifter model requirements
import neureye.models.encoders as encoders
import neureye.models.cores as cores
import neureye.models.layers as layers
import neureye.models.readouts as readouts
import neureye.models.regularizers as regularizers
import neureye.models.utils as ut

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dtype = torch.float32

# Where saved models and checkpoints go -- this is to be automated
print( 'Save_dir =', dirname)

#%% Fixation dataset
sess_list = ['20200304'] # multiple datasets can be loaded
stimlist = ['Gabor', 'BackImage', 'Grating', 'Dots', 'FixRsvpStim']
num_lags = 18
downsample_t = 1

train_ds = datasets.FixationMultiDataset(sess_list, datadir,
    requested_stims=stimlist,
    stimset='Train',
    downsample_t=downsample_t,
    max_fix_length=2000,
    saccade_basis={'max_len': 40, 'num':15},
    num_lags = num_lags)

test_ds = datasets.FixationMultiDataset(sess_list, datadir,
    requested_stims=['Gabor'],
    stimset='Test',
    downsample_t=downsample_t,
    max_fix_length=2000,
    saccade_basis={'max_len': 40, 'num':15},
    num_lags = num_lags)

#%% Load Gabor / Dots data to check STAS
data = train_ds[train_ds.get_stim_indices(['Gabor', 'Dots'])]

#%% helper function for plotting STAS
def plot_stas(stas):
    
    NC = stas.shape[-1]
    num_lags= stas.shape[0]

    sx = int(np.ceil(np.sqrt(NC*2)))
    sy = int(np.round(np.sqrt(NC*2)))
    mod2 = sy % 2
    sy += mod2
    sx -= mod2
    mu = np.zeros((NC,2))

    plt.figure(figsize=(sx*2,sy*2))
    for cc in range(NC):
        w = stas[:,:,:,cc]

        wt = np.std(w, axis=0)
        wt /= np.max(np.abs(wt)) # normalize for numerical stability
        # softmax
        wt = wt**10
        wt /= np.sum(wt)
        sz = wt.shape
        xx,yy = np.meshgrid(np.linspace(-1, 1, sz[1]), np.linspace(1, -1, sz[0]))

        mu[cc,0] = np.minimum(np.maximum(np.sum(xx*wt), -.5), .5) # center of mass after softmax
        mu[cc,1] = np.minimum(np.maximum(np.sum(yy*wt), -.5), .5) # center of mass after softmax

        w = (w -np.mean(w) )/ np.std(w)

        bestlag = np.argmax(np.std(w.reshape( (num_lags, -1)), axis=1))
        plt.subplot(sx,sy, cc*2 + 1)
        v = np.max(np.abs(w))
        plt.imshow(w[bestlag,:,:], aspect='auto', interpolation=None, vmin=-v, vmax=v, cmap="coolwarm", extent=(-1,1,-1,1))
        plt.title(cc)
        plt.subplot(sx,sy, cc*2 + 2)
        i,j=np.where(w[bestlag,:,:]==np.max(w[bestlag,:,:]))
        t1 = w[:,i[0], j[0]]
        plt.plot(t1, '-ob')
        i,j=np.where(w[bestlag,:,:]==np.min(w[bestlag,:,:]))
        t2 = w[:,i[0], j[0]]
        plt.plot(t2, '-or')
    
    return mu

# compute STAs
def get_stas(stim, robs, dfs, num_lags, dims, NC):
    stas = torch.zeros([num_lags] + dims + [NC])
    for lag in range(num_lags):
        if lag == 0:
            sta = stim.T@(robs*dfs)
        else:
            sta = stim[:-lag,:].T@(robs[lag:,:]*dfs[lag:,:])

        stas[lag,:,:,:] = sta.reshape(dims + [NC])
    
    return stas

#%% get and plot STAS
stas0 = get_stas(data['stim'], data['robs'], data['dfs'], num_lags, train_ds.dims[1:], train_ds.NC)
mu = plot_stas(stas0.numpy())
plt.show()
#%% show RF positions and mean firing rates
plt.figure()
plt.plot(mu[:,0], mu[:,1], 'ob')

rbase = data['robs'].mean(dim=0)
plt.figure()
plt.plot(rbase)
plt.show()
#%% smaller dataset to work with
data = train_ds[0:2]

#%%


import torch.nn as nn
class STconvModel(nn.Module):
    def __init__(
        self,
        input_dims,
        kernel_size,
        out_features,
        tent_spacing=1,
        reg_types=["d2x", "center"],
        reg_amt=[.00005,.005]):

        super(STconvModel, self).__init__()

        reg_size = list(kernel_size)
        regularizer_config = {'dims': [reg_size[1]//tent_spacing] + reg_size[2:],
                            'type': reg_types, 'amount': reg_amt}

        self._input_weights_regularizer = regularizers.RegMats(**regularizer_config)

        self.features = nn.Sequential()
        layer = nn.Sequential()
        layer.add_module('conv', nn.utils.weight_norm(layers.STconv( input_dims= input_dims,
            kernel_size=kernel_size,
            out_features=out_features,
            tent_spacing=tent_spacing),
            dim=0, name='weight'))

        layer.add_module('nonlin', layers.AdaptiveELU(0,1.0))
        layer.add_module('norm', layers.divNorm(out_features))

        self.features.add_module("layer0", layer)

        self.outchannels = out_features

    def input_reg(self):
        return self._input_weights_regularizer(self.features[0].conv.weight.squeeze())

    def regularizer(self):
        return self.input_reg()

    def forward(self, x):
        return self.features(x)

    
#%% build model
from neureye.models.trainers import Trainer, EarlyStopping
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
from neureye.models.losses import PoissonNLLDatFilterLoss

def shifter_model(hidden_channels=16,
    input_kern = 19):

    core = STconvModel(train_ds.dims,
        (1,num_lags, input_kern,input_kern),
        hidden_channels,
        tent_spacing=2,
        reg_types=["d2x", "center", "d2t"],
        reg_amt=[.000005, .01, 0.00001]
    )

    lengthscale = 1.0

    # initialize input layer to be centered
    regw = regularizers.gaussian2d(input_kern,sigma=input_kern//4)
    core.features[0].conv.weight.data = torch.einsum('icjkm,km->icjkm', core.features[0].conv.weight.data, torch.tensor(regw))
        
    # Readout
    in_shape = [core.outchannels] + train_ds.dims[1:]
    bias = True
    readout = readouts.Point2DGaussian(in_shape, train_ds.NC, bias, init_mu_range=0.1, init_sigma=1, batch_sample=True,
                    gamma_l1=0,gamma_l2=0.00001,
                    align_corners=True, gauss_type='uncorrelated',
                    constrain_positive=False,
                    shifter= {'hidden_features': 20,
                            'hidden_layers': 1,
                            'final_tanh': False,
                            'activation': "softplus",
                            'lengthscale': lengthscale}
                            )

    modifiers = {'stimlist': ['saccade'],
                'gain': [data['saccade'].shape[1]],
                'offset':[data['saccade'].shape[1]],
                'stage': "readout",
                'outdims': train_ds.NC}

    # combine core and readout into model
    model = encoders.EncoderMod(core, readout, modifiers=modifiers, loss=PoissonNLLDatFilterLoss(log_input=True, reduction='mean'))

    # initialize readout based on spike rate and STA centers
    model.readout.bias.data = rbase
    model.readout._mu.data[0,:,0,:] = torch.tensor(mu.astype('float32')) # initiaalize mus

    return model

def train_model(model, save_path, version,
        batchsize = 15,
        weight_decay=.01,
        learning_rate=.01, # high initial learning rate because we decay on plateau
        betas=[.9, .999],
        amsgrad=False,
        early_stopping_patience=4,
        seed=None):

    import time
    # get data
    train_dl, valid_dl = ut.get_dataloaders(train_ds, batch_size=batchsize)

    # optimizer
    optimizer = torch.optim.AdamW(model.parameters(),
        lr=learning_rate,
        betas=betas,
        weight_decay=weight_decay,
        amsgrad=amsgrad)

    earlystopping = EarlyStopping(patience=early_stopping_patience,delta=0.0)

    trainer = Trainer(model, optimizer,
                early_stopping=earlystopping,
                dirpath=save_path,
                optimize_graph=False,
                scheduler=None,
                max_epochs=100,
                version=version)

    t0 = time.time()
    trainer.fit( model, train_dl, valid_dl, seed=seed)
    t1 = time.time()

    print('  Fit complete:', t1-t0, 'sec elapsed')

# Plot utilities for shifter
def plot_shifter(shifter, valid_eye_rad=5.2, ngrid = 100):
    xx,yy = np.meshgrid(np.linspace(-valid_eye_rad, valid_eye_rad,ngrid),np.linspace(-valid_eye_rad, valid_eye_rad,ngrid))
    xgrid = torch.tensor( xx.astype('float32').reshape( (-1,1)))
    ygrid = torch.tensor( yy.astype('float32').reshape( (-1,1)))

    inputs = torch.cat( (xgrid,ygrid), dim=1)

    xyshift = shifter(inputs).detach().numpy()

    xyshift/=valid_eye_rad/60 # conver to arcmin
    vmin = np.min(xyshift)
    vmax = np.max(xyshift)

    shift = [xyshift[:,0].reshape((ngrid,ngrid))]
    shift.append(xyshift[:,1].reshape((ngrid,ngrid))) 
    plt.figure(figsize=(6,3))
    plt.subplot(1,2,1)
    plt.imshow(shift[0], extent=(-valid_eye_rad,valid_eye_rad,-valid_eye_rad,valid_eye_rad), interpolation=None, vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.subplot(1,2,2)
    plt.imshow(shift[1], extent=(-valid_eye_rad,valid_eye_rad,-valid_eye_rad,valid_eye_rad), interpolation=None, vmin=vmin, vmax=vmax)
    plt.colorbar()

    return shift

def shift_stim(self, im, eyepos):
        """
        apply shifter to translate stimulus as a function of the eye position
        """
        import torch.nn.functional as F
        import torch
        affine_trans = torch.tensor([[[1., 0., 0.], [0., 1., 0.]]])
        sz = im.shape
        eyepos = torch.tensor(eyepos.astype('float32'))
        im = torch.tensor(im[:,None,:,:].astype('float32'))
        im = im.permute((3,1,0,2))

        shift = self.shifter(eyepos).detach()
        aff = torch.tensor([[1,0,0],[0,1,0]])

        affine_trans = shift[:,:,None]+aff[None,:,:]
        affine_trans[:,0,0] = 1
        affine_trans[:,0,1] = 0
        affine_trans[:,1,0] = 0
        affine_trans[:,1,1] = 1

        n = im.shape[0]
        grid = F.affine_grid(affine_trans, torch.Size((n, 1, sz[0], sz[1])), align_corners=True)

        im2 = F.grid_sample(im, grid, align_corners=True)
        im2 = im2[:,0,:,:].permute((1,2,0)).detach().cpu().numpy()

        return im2

#%% Do fitting

model = shifter_model()

out = model.training_step(data)
#%%
for version in [7]:
    
    model = shifter_model()
    # model.readout.shifter.requires_grad_ = False

    save_path = os.path.join(dirname, 'jake_trainer')

    train_model(model, save_path, version, early_stopping_patience=10)


#%%
# model2 = model
# #%% Load model after fit
save_path = os.path.join(dirname, 'jake_trainer')
version = 5
outdict = ut.get_fit_versions(save_path)
if version is None:
    version = outdict['version_num'][np.argmin(outdict['val_loss'])]
    print("No version requested. Best version is %d" %version)

vind = np.where(np.asarray(outdict['version_num']) == version)[0][0]
mod_path = outdict['model_file'][vind]

model2 = torch.load(mod_path)
# model2 = encoders.EncoderMod.load_from_checkpoint(modpath)
torch.nn.utils.remove_weight_norm(model2.core.features[0].conv)

# Plot filters
# model2.core.plot_filters()

#%%
w = model2.core.features[0].conv.weight.detach().cpu()
w = w.squeeze()
w = torch.einsum('ntwh, tz -> nzwh', w, model2.core.features[0].conv.tent_basis.cpu())
w = w.permute((1,2,3,0))
_ = plot_stas(w.numpy())
plt.show()

#%% Plot shifter

shifter = model2.readout.shifter.cpu()
shift = plot_shifter(shifter)
plt.show()


#%% compute shift from shifter

shift = []
for i in range(len(train_ds.eyepos)):
    shift.append(shifter(torch.tensor(train_ds.eyepos[i], dtype=torch.float32)).detach())

#%% recalc stas
train_ds.shift = shift

data_shift = train_ds[train_ds.get_stim_indices(['Gabor', 'Dots'])]

#%%
stas1 = get_stas(data_shift['stim'], data_shift['robs'], data_shift['dfs'], num_lags*2, train_ds.dims[1:], train_ds.NC)
_ = plot_stas(stas1.numpy())
plt.show()
_ = plot_stas(stas0.numpy())
plt.show()

#%% Try loading older model fit
save_dir='./checkpoints/v1calibration_ls{}'.format(1)

save_path = os.path.join(save_dir, 'jake_trainer')
version = None
outdict = ut.get_fit_versions(save_path)
if version is None:
    version = outdict['version_num'][np.argmin(outdict['val_loss'])]
    print("No version requested. Best version is %d" %version)

vind = np.where(np.asarray(outdict['version_num']) == version)[0][0]
mod_path = outdict['model_file'][vind]

model2 = torch.load(mod_path)
# model2 = encoders.EncoderMod.load_from_checkpoint(modpath)
torch.nn.utils.remove_weight_norm(model2.core.features[0].conv)

# Plot filters
model2.core.plot_filters()
plt.show()

# %% reload data with shifter
data_path='/home/jake/Datasets/Mitchell/stim_movies/'
gd_shift = datasets.PixelDataset(sess_list[0], stims=['Gabor', 'Dots'],
    stimset="Train", num_lags=num_lags,
    downsample_t=2,
    downsample_s=1,
    valid_eye_rad=5.2,
    dirname=data_path,
    shifter=shifter,    
    dim_order='txy',
    include_eyepos=True,
    flatten=False,  # flattens data, False for this model
    download=False, # download dataset from server if not already downloaded
    temporal=False, # flag for backwards compatiability (must be false)
    preload=False) # turn preload on to speed up fitting (requires lots of CPU memory)

#%% compute STAs with shifted stimulus
index = np.where(gd_shift.stim_indices==0)[0]
sta_shift = get_stas(gd_shift, index)

#%% Plot STAS
mu = plot_stas(sta_shift)
# %%
