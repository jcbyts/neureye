import torch
from torch import nn
from torch.nn import functional as F
# import regularizers
from neureye.models.readouts import Readout
from neureye.models.cores import Core
from neureye.models.losses import PoissonNLLDatFilterLoss
from neureye.models.utils import save_hyperparameters, assemble_ffnetworks

"""
Main encoder class
"""
class Encoder(nn.Module):
    
    def __init__(self,
        core=Core(),
        readout=Readout(),
        output_nl=nn.Softplus(),
        loss=PoissonNLLDatFilterLoss(log_input=False, reduction='mean'),
        val_loss=None,
        detach_core=False,
        data_dir='',
        **kwargs):

        super(Encoder, self).__init__()
        self.core = core
        self.readout = readout
        self.detach_core = detach_core

        self.hparams = save_hyperparameters()
        
        if val_loss is None:
            self.val_loss = loss
        else:
            self.val_loss = val_loss

        self.output_nl = output_nl
        self.loss = loss

    def forward(self, x, shifter=None, sample=None):
        x = self.core(x)
        if self.detach_core:
            x = x.detach()
        if "shifter" in dir(self.readout) and self.readout.shifter and shifter is not None:
            x = self.readout(x, shift=self.readout.shifter(shifter))
        else:
            x = self.readout(x)

        return self.output_nl(x)

    def training_step(self, batch, batch_idx=None):
        x = batch['stim']
        y = batch['robs']
        if "shifter" in dir(self.readout) and batch['eyepos'] is not None and self.readout.shifter:
            y_hat = self(x, shifter=batch['eyepos'])
        else:
            y_hat = self(x)
        
        if 'dfs' in batch.keys():
            loss = self.loss(y_hat, y, batch['dfs'])
        else:
            loss = self.loss(y_hat, y)
        
        regularizers = int(not self.detach_core) * self.core.regularizer() + self.readout.regularizer()

        return {'loss': loss + regularizers, 'train_loss': loss, 'reg_loss': regularizers}

    def validation_step(self, batch, batch_idx=None):

        x = batch['stim']
        y = batch['robs']
        if "shifter" in dir(self.readout) and batch['eyepos'] is not None and self.readout.shifter:
            y_hat = self(x, shifter=batch['eyepos'])
        else:
            y_hat = self(x)
        
        if 'dfs' in batch.keys():
            loss = self.val_loss(y_hat, y, batch['dfs'])
        else:
            loss = self.val_loss(y_hat, y)

        return {'loss': loss, 'val_loss': loss}
    
    def on_save_checkpoint(self, checkpoint):
        # track the core, readout, shifter class and state_dicts
        checkpoint['core_type'] = type(self.core)
        checkpoint['core_hparams'] = self.core.hparams
        checkpoint['core_state_dict'] = self.core.state_dict()

        checkpoint['readout_type'] = type(self.readout)
        checkpoint['readout_hparams'] = self.readout.hparams
        checkpoint['readout_state_dict'] = self.readout.state_dict()

    def on_load_checkpoint(self, checkpoint):
        # properly handle core, readout, shifter state_dicts
        self.core = checkpoint['core_type'](**checkpoint['core_hparams'])
        self.readout = checkpoint['readout_type'](**checkpoint['readout_hparams'])
        self.core.load_state_dict(checkpoint['core_state_dict'])
        self.readout.load_state_dict(checkpoint['readout_state_dict'])

class EncoderMod(nn.Module): # IN PROGRESS
    def __init__(self,
        core=Core(),
        readout=Readout(),
        output_nl=nn.Softplus(),
        modifiers=None,
        gamma_mod=.1,
        loss=nn.PoissonNLLLoss(log_input=False, reduction='mean'),
        val_loss=None,
        detach_core=False):

        super().__init__()
        self.core = core
        self.readout = readout
        self.detach_core = detach_core
        self.hparams = save_hyperparameters()
        
        # initialize variables for modifier: these all need to be here regardless of whether the modifiers are used so we can load the model checkpoints
        self.offsets = nn.ModuleList()
        self.gains = nn.ModuleList()
        self.offsetstims = []
        self.gainstims = []
        self.modify = False
        self.register_buffer("offval", torch.zeros(1))
        self.register_buffer("gainval", torch.ones(1))

        if self.hparams.modifiers is not None:
            """
            modifier is a hacky addition to the model to allow for offsets and gains at a certain stage in the model
            The default stage is after the readout
            example modifier input:
            modifier = {'stimlist': ['frametent', 'saccadeonset'],
            'gain': [40, None],
            'offset':[40,20],
            'stage': "readout",
            'outdims: gd.NC}
            """
            if type(self.hparams.modifiers)==dict:
                self.modify = True

                nmods = len(self.hparams.modifiers['stimlist'])
                assert nmods==len(self.hparams.modifiers["offset"]), "Encoder: modifier specified incorrectly"
                
                if 'stage' not in self.hparams.modifiers.keys():
                    self.hparams.modifiers['stage'] = "readout"
                
                # set the output dims (this hast to match either the readout output the whole core is modulated)
                if self.hparams.modifiers['stage']=="readout":
                    outdims = self.hparams.modifiers['outdims']
                elif self.hparams.modifiers['stage']=="core":
                    outdims = 1

                self.modifierstage = self.hparams.modifiers["stage"]
                for imod in range(nmods):
                    if self.hparams.modifiers["offset"][imod] is not None:
                        self.offsetstims.append(self.hparams.modifiers['stimlist'][imod])
                        self.offsets.append(nn.Linear(self.hparams.modifiers["offset"][imod], outdims, bias=False))
                    if self.hparams.modifiers["gain"][imod] is not None:
                        self.gainstims.append(self.hparams.modifiers['stimlist'][imod])
                        self.gains.append(nn.Linear(self.hparams.modifiers["gain"][imod], outdims, bias=False))
        else:
            self.modify = False

        if val_loss is None:
            self.val_loss = loss
        else:
            self.val_loss = val_loss

        self.output_nl = output_nl
        self.loss = loss

    def forward(self, x, shifter=None, sample=None):
        dlist = dir(self)
        if "offsets" in dlist:
            use_offsets = True
        else:
            use_offsets = False

        if "gains" in dlist:
            use_gains = True
        else:
            use_gains = False

        offset = self.offval
        if use_offsets:
            for offmod,stim in zip(self.offsets, self.offsetstims):
                offset = offset + offmod(sample[stim])

        gain = self.gainval
        if use_gains:
            for gainmod,stim in zip(self.gains, self.gainstims):
                gain = gain * (self.gainval + gainmod(sample[stim]))

        if self.modify and self.modifierstage=="stim":
            x *= gain
            x += offset

        x = self.core(x)

        if self.detach_core:
            x = x.detach()
        
        if self.modify and self.modifierstage=="core":
            x *= gain
            x += offset

        if "shifter" in dir(self.readout) and self.readout.shifter and shifter is not None:
            shift = self.readout.shifter(shifter)
        else:
            shift = None

        x = self.readout(x, shift=shift)

        if self.modify and self.modifierstage=="readout":
            x *= gain
            x += offset

        return self.output_nl(x)

    def training_step(self, batch, batch_idx=None):
        x = batch['stim']
        y = batch['robs']
        if "shifter" in dir(self.readout) and batch['eyepos'] is not None and self.readout.shifter:
            shift = batch['eyepos']
        else:
            shift = None

        if self.modify:
            y_hat = self(x, shifter=shift, sample=batch)
        else:
            y_hat = self(x, shifter=shift)
        
        if 'dfs' in batch.keys():
            loss = self.loss(y_hat, y, batch['dfs'])
        else:
            loss = self.loss(y_hat, y)

        regularizers = int(not self.detach_core) * self.core.regularizer() + self.readout.regularizer()
        # regularizers for modifiers
        reg = 0
        if self.modify:
            for imod in range(len(self.offsets)):
                reg += self.offsets[imod].weight.pow(2).sum().sqrt()
            for imod in range(len(self.gains)):
                reg += self.offsets[imod].weight.pow(2).sum().sqrt()

        regularizers = regularizers + self.hparams.gamma_mod * reg
        # self.log('train_loss', loss, 'reg_pen', regularizers + self.hparams.gamma_mod * reg)
        return {'loss': loss + regularizers, 'train_loss': loss, 'reg_loss': regularizers}
    
    def validation_step(self, batch, batch_idx=None):

        x = batch['stim']
        y = batch['robs']
        if "shifter" in dir(self.readout) and batch['eyepos'] is not None and self.readout.shifter:
            shift = batch['eyepos']
        else:
            shift = None

        if self.modify:
            y_hat = self(x, shifter=shift, sample=batch)
        else:
            y_hat = self(x, shifter=shift)
        

        if 'dfs' in batch.keys():
            loss = self.loss(y_hat, y, batch['dfs'])
        else:
            loss = self.loss(y_hat, y)

        return {'loss': loss, 'val_loss': loss}
    
    def on_save_checkpoint(self, checkpoint):
        # track the core, readout, shifter class and state_dicts
        checkpoint['core_type'] = type(self.core)
        checkpoint['core_hparams'] = self.core.hparams
        checkpoint['core_state_dict'] = self.core.state_dict()

        checkpoint['readout_type'] = type(self.readout)
        checkpoint['readout_hparams'] = self.readout.hparams
        checkpoint['readout_state_dict'] = self.readout.state_dict() # TODO: is this necessary or included in self state_dict?

    def on_load_checkpoint(self, checkpoint):
        # properly handle core, readout, shifter state_dicts
        self.core = checkpoint['core_type'](**checkpoint['core_hparams'])
        self.readout = checkpoint['readout_type'](**checkpoint['readout_hparams'])
        self.core.load_state_dict(checkpoint['core_state_dict'])
        self.readout.load_state_dict(checkpoint['readout_state_dict'])