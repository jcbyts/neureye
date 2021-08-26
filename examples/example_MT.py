#%% 
import sys
# sys.path.insert(0, '/home/jake/Repos/')
sys.path.insert(0, '/home/jake/Data/Repos/')

import os

import numpy as np
import torch
import torch.nn as nn

import matplotlib.pyplot as plt  # plotting
import seaborn as sns

# the dataset
import datasets.mitchell.mtdots as datasets

# shifter model requirements
import neureye.models.regularizers as regularizers
import neureye.models.utils as ut

#%%
import importlib
importlib.reload(datasets)

# %% Load data
sess = '20190120'
data_dir = '/home/jake/Data/Datasets/MitchellV1FreeViewing/MT_RF/'
mt_ds = datasets.MTDotsDataset(sess, data_dir)

sample = mt_ds[:10]

#%% Model setup
class GLM(nn.Module):
    def __init__(self, dims, NC, bias=True, gamma=1e-3, gamma_l1=1e-6):
        super(GLM, self).__init__()

        self.dims = dims
        self.NI = np.prod(np.asarray(dims))
        self.NC = NC
        self.loss = nn.PoissonNLLLoss(log_input=False)
        self.features = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.NI, NC, bias=bias),
            nn.Softplus()
        )
        self.reg = regularizers.RegMats(dims=[dims[3], dims[1], dims[2]], type=['d2x', 'd2t'], amount=[.5, 1])
        self.gamma = gamma
        self.gamma_l1 = gamma_l1
        self.initialize()

    def forward(self, x):
        return self.features(x)

    def training_step(self, batch):
        stim = batch['stim']
        robs = batch['robs']
        loss = self.loss(self.features(stim), robs)
        w = self.features[1].weight
        w = w.view([self.NC * self.dims[0]] + self.dims[1:]).permute((0, 3, 1 , 2))
        reg = self.gamma*self.reg(w) + self.gamma_l1*w.abs().mean()
        return {'loss': loss+reg, 'train_loss': loss, 'reg': reg}
    
    def validation_step(self, batch):
        stim = batch['stim']
        robs = batch['robs']
        loss = self.loss(self.features(stim), robs)
        return {'loss': loss, 'val_loss': loss}

    def initialize(self):
        self.apply(self.init_)

    def fit(self, ds, batch_size=500, max_epochs=100, version=None, save_path=os.path.join('.', 'checkpoints', 'mt_glm')):
        from torch.utils.data import DataLoader, random_split
        import neureye.models.trainers as trainers

        lossfun = nn.PoissonNLLLoss(log_input=False, reduction='none')

        ntrain = int(0.8*len(ds))
        nval = len(ds)-ntrain

        train_ds, val_ds = random_split(ds, [ntrain, nval])

        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.LBFGS(glm.parameters(), history_size=10,
                                max_iter=4,
                                line_search_fn="strong_wolfe")

        early_stopping = trainers.EarlyStopping(patience=1)

        trainer = trainers.Trainer(glm, optimizer=optimizer,
            early_stopping=early_stopping, version=version,
            dirpath=save_path,
            max_epochs=max_epochs, optimize_graph=True)

        trainer.fit(glm, train_dl, val_dl, seed=666)

        sample = val_ds[:]
        self.to('cpu')
        yhat = self(sample['stim'])
        val_loss = lossfun(yhat, sample['robs']).mean(dim=0)

        return val_loss.detach().numpy()

    @staticmethod
    def init_(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0)


# %% Train model
dims = [mt_ds.num_channels, mt_ds.NX, mt_ds.NY, mt_ds.num_lags]
batch_size = 10000 # Should be big (using L-BFGS), but set for your GPU size
save_path = os.path.join('.', 'checkpoints', 'mt_glm')

glm = GLM(dims, mt_ds.NC, gamma=2e-5, gamma_l1=1e-4) # glm with d2xt and l1
val_loss = glm.fit(mt_ds, batch_size=batch_size, version=None, save_path=save_path)

glm.gamma_l1=1e-1 # step up L1 penalty, see if it cleans up
val_loss = glm.fit(mt_ds, batch_size=batch_size, version=None, save_path=save_path)

#%% plot
wtsAll = glm.features[1].weight.detach().cpu().numpy().T
    

#%% plot example cells
# 16
cids = [0, 1, 4, 9, 12,13,15, 17, 19]

plt.figure(figsize=(10,10))
for ii,cc in enumerate(cids):
    rf = mt_ds.get_rf(wtsAll, cc)

    plt.subplot(3,3,ii+1)
    
    xx = np.meshgrid(mt_ds.xax, mt_ds.yax)
    dx = rf['dx']
    dy = rf['dy']
    amp = rf['amp']
    plt.quiver(xx[0]-np.mean(xx[0]), xx[1]-np.mean(xx[1]), dx/np.max(amp), dy/np.max(amp), amp,
                pivot='tail',units='width', width=.01,
                cmap=plt.cm.coolwarm,
                scale=10, headwidth=2, headlength=1.5)

    plt.xlim((-15,15))
    plt.ylim((-15,15))
    if ii % 3 != 0:
        plt.yticks([])
    
    if ii < 6:
        plt.xticks([])

    plt.axhline(0, color='k')
    plt.axvline(0, color='k')

plt.savefig('./figures/examples_spatial.pdf')
plt.show()

#%% plot temporal kernels
plt.figure(figsize=(10,10))
for ii,cc in enumerate(cids):
    rf = mt_ds.get_rf(wtsAll, cc)

    plt.subplot(3,3,ii+1)
    
    plt.plot(rf['lags'], rf['tpeak'], '-o', color=plt.cm.coolwarm(np.inf))
    plt.plot(rf['lags'], rf['tmin'], '-o', color=plt.cm.coolwarm(-np.inf))
    plt.axhline(0, color='k')

    if ii % 3 != 0:
        plt.yticks([])
    
    if ii < 6:
        plt.xticks([])

    plt.axhline(0, color='k')
    
plt.savefig('./figures/examples_temporal.pdf')
plt.show()
#%%
NX = mt_ds.NX
NY = mt_ds.NY
num_lags = mt_ds.num_lags
xx = np.meshgrid(mt_ds.xax, mt_ds.yax)

plt.figure(figsize=(10, glm.NC*2))

for cc in range(glm.NC):

    rf = mt_ds.get_rf(wtsAll, cc)
    dx = rf['dx']
    dy = rf['dy']
    amp = rf['amp']

    ax = plt.subplot(glm.NC,3,cc*3+1)

    plt.quiver(xx[0]-np.mean(xx[0]), xx[1]-np.mean(xx[1]), dx/np.max(amp), dy/np.max(amp),
            amp, cmap=plt.cm.coolwarm,
            pivot='tail',units='width', width=.008,
            scale=10, headwidth=2.5, headlength=2.5)


    plt.axhline(0, color='gray', )
    plt.axvline(0, color='gray')

    plt.xlabel('Azimuth (d.v.a.)')
    plt.ylabel('Elevation (d.v.a)')

    plt.xticks(np.arange(-15,18,5))

    ax2 = plt.subplot(glm.NC,3,cc*3+2)

    plt.plot(rf['lags'], rf['tpeak'], '-o', color=plt.cm.coolwarm(np.inf), ms=3)
    plt.plot(rf['lags'], rf['tmin'], '-o', color=plt.cm.coolwarm(-np.inf), ms=3)
    plt.xlabel('Lags (ms)')
    plt.ylabel('Power (along preferred direction)')

    plt.axhline(0, color='gray')

    ax3 = plt.subplot(glm.NC,3,cc*3+3)
    tc = mt_ds.plot_tuning_curve(cc, rf['amp'])

plt.show()
#%% Example cell
# import seaborn as sns

cc = 0
rf = mt_ds.get_rf(wtsAll, cc)

dx = rf['dx']
dy = rf['dy']
amp = rf['amp']

plt.figure()
plt.quiver(xx[0]-np.mean(xx[0]), xx[1]-np.mean(xx[1]), dx/np.max(amp), dy/np.max(amp),
            amp, cmap=plt.cm.coolwarm,
            pivot='tail',units='width', width=.008,
            scale=15, headwidth=2.5, headlength=2.5)


plt.axhline(0, color='gray', )
plt.axvline(0, color='gray')

plt.xlabel('Azimuth (d.v.a.)')
plt.ylabel('Elevation (d.v.a)')
# plt.show()
plt.savefig('./figures/example_spatial_{}.pdf'.format(cc))

plt.figure()
plt.plot(rf['lags'], rf['tpeak'], '-o', color=plt.cm.coolwarm(np.inf), ms=3)
plt.plot(rf['lags'], rf['tmin'], '-o', color=plt.cm.coolwarm(-np.inf), ms=3)
plt.xlabel('Lags (ms)')
plt.ylabel('Power (along preferred direction)')

plt.axhline(0, color='gray')
sns.despine(offset=0, trim=True)
# plt.show()
plt.savefig('./figures/example_temporal_{}.pdf'.format(cc))

plt.figure()
tc = mt_ds.plot_tuning_curve(cc, rf['amp'])
plt.ylim((0, 40))
sns.despine(offset=0, trim=True)
# plt.show()
plt.savefig('./figures/example_tuning_{}.pdf'.format(cc))

plt.show()
# %%
