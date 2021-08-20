import os
import shutil
import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy
from collections import OrderedDict

def get_trainer(dataset,
        version=1,
        save_dir='./checkpoints',
        name='jnkname',
        auto_lr=False,
        batchsize=1000,
        earlystopping=True,
        earlystoppingpatience=10,
        max_epochs=150,
        num_workers=1,
        gradient_clip_val=0,
        seed=None):
    """
    Returns a pytorch lightning trainer and splits the training set into "train" and "valid"
    """
    from torch.utils.data import Dataset, DataLoader, random_split
    from pytorch_lightning import Trainer, seed_everything
    from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
    from pytorch_lightning.loggers import TestTubeLogger
    from pathlib import Path

    
    save_dir = Path(save_dir)
    n_val = np.floor(len(dataset)/5).astype(int)
    n_train = (len(dataset)-n_val).astype(int)

    gd_train, gd_val = random_split(dataset, lengths=[n_train, n_val])

    # build dataloaders
    train_dl = DataLoader(gd_train, batch_size=batchsize, num_workers=num_workers, pin_memory=True)
    valid_dl = DataLoader(gd_val, batch_size=batchsize, num_workers=num_workers, pin_memory=True)

    # Train
    if earlystopping:
        early_stop_callback = EarlyStopping(monitor='val_loss', min_delta=0.0, patience=earlystoppingpatience)
    # checkpoint_callback = ModelCheckpoint(monitor='val_loss')

    # logger = TestTubeLogger(
    #     save_dir=save_dir,
    #     name=name,
    #     version=version  # fixed to one to ensure checkpoint load
    # )

    trainer = Trainer(gpus=1, progress_bar_refresh_rate=20,
        default_root_dir=save_dir,
        max_epochs=max_epochs,
        callbacks=[early_stop_callback],
        auto_lr_find=auto_lr)

    # # ckpt_folder = save_dir / sessid / 'version_{}'.format(version) / 'checkpoints'
    # if earlystopping:
    #     trainer = Trainer(gpus=1, callbacks=[early_stop_callback],
    #         checkpoint_callback=checkpoint_callback,
    #         logger=logger,
    #         deterministic=False,
    #         gradient_clip_val=gradient_clip_val,
    #         accumulate_grad_batches=1,
    #         progress_bar_refresh_rate=20,
    #         max_epochs=1000,
    #         auto_lr_find=auto_lr)
    # else:
    #     trainer = Trainer(gpus=1,
    #         checkpoint_callback=checkpoint_callback,
    #         logger=logger,
    #         deterministic=False,
    #         gradient_clip_val=gradient_clip_val,
    #         accumulate_grad_batches=1,
    #         progress_bar_refresh_rate=20,
    #         max_epochs=max_epochs,
    #         auto_lr_find=auto_lr)

    if seed:
        seed_everything(seed)

    return trainer, train_dl, valid_dl

def get_fit_versions(save_dir):
    '''
        Find versions of the fit model
        Arguments:
            save_dir: directory where the logging happens
    '''

    import re
    from tensorboard.backend.event_processing import event_accumulator

    data_dir = os.path.join(save_dir, 'lightning_logs')
    dirlist = [x for x in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, x))]
            
    versionlist = [re.findall('(?!version_)\d+', x) for x in dirlist]
    versionlist = [int(x[0]) for x in versionlist if not not x]
    outdict = {'version_num': [],
            'events_file': [],
            'check_file': [],
            'val_loss': [],
            'val_loss_steps': []}

    for v in versionlist:
        # events_file = os.path.join(data_dir, model_name, 'version_%d' %v, 'events.out.tfevents.%d' %v)
        vpath = os.path.join(data_dir, 'version_%d' %v)
        vplist = os.listdir(vpath)

        tfeventsfiles = [x for x in vplist if 'events.out.tfevents' in x]
        checkpath = os.path.join(vpath, 'checkpoints')
        checkfiles = [x for x in os.listdir(checkpath) if 'ckpt' in x]

        if len(tfeventsfiles) == 1:
            evfile = os.path.join(vpath, tfeventsfiles[0])
            # read from tensorboard backend
            ea = event_accumulator.EventAccumulator(evfile)
            ea.Reload()
            try:
                val = np.asarray([x.value for x in ea.scalars.Items("val_loss")])
                bestval = np.min(val)

                outdict['version_num'].append(v)
                outdict['events_file'].append(evfile)
                outdict['check_file'].append(os.path.join(checkpath, checkfiles[-1]))
                outdict['val_loss_steps'].append(val)
                outdict['val_loss'].append(bestval)
            except:
                print("error")
                continue
    return outdict


def find_best_epoch(ckpt_folder):
    # from os import listdir
    # import glob
    """
    Find the highest epoch in the Test Tube file structure.
    :param ckpt_folder: dir where the checpoints are being saved.
    :return: Integer of the highest epoch reached by the checkpoints.
    """
    try:
        # ckpt_files = listdir(ckpt_folder)  # list of strings
        ckpt_files = list(ckpt_folder.glob('*.ckpt'))
        epochs = []
        fname = []
        for f in ckpt_files:
            n = f.name
            fname.append(n)
            start = n.find("=")
            step = n.find("step")
            if step < 0:
                epochs.append(int(n[start+1:-5]))
            else:
                epochs.append(int(n[start+1:step-1]))

        out = max(epochs)
        fn = [fname[i] for i in range(len(epochs)) if epochs[i]==out]
        out = fn[0]
    except FileNotFoundError:
        out = None
    return out

def get_null_adjusted_ll(model, sample, bits=False):
    '''
    get null-adjusted log likelihood
    bits=True will return in units of bits/spike
    '''
    m0 = model.cpu()
    loss = torch.nn.PoissonNLLLoss(log_input=False, reduction='none')
    lnull = -loss(torch.ones(sample['robs'].shape)*sample['robs'].mean(axis=0), sample['robs']).detach().cpu().numpy().sum(axis=0)
    yhat = m0(sample['stim'], shifter=sample['eyepos'])
    llneuron = -loss(yhat,sample['robs']).detach().cpu().numpy().sum(axis=0)
    rbar = sample['robs'].sum(axis=0).numpy()
    ll = (llneuron - lnull)/rbar
    if bits:
        ll/=np.log(2)
    return ll