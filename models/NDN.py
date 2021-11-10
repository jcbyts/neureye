import torch
from torch import nn
from torch.nn import functional as F

import neureye.models.layers as layers
import neureye.models.losses as losses
from neureye.models.utils import save_hyperparameters, assemble_ffnetworks

class NDN(nn.Module):

    def __init__(self,
        ffnet_list=None,
        opt_params=None,
        model_name='NDN_model',
        data_dir='./checkpoints',
        **kwargs):

        super(NDN, self).__init__(**kwargs)

        self.hparams = save_hyperparameters() # hyper parameters will get logged

        if ffnet_list is not None:
            self.networks = assemble_ffnetworks(ffnet_list)

        self.model_name = model_name
        self.data_dir = data_dir
        
    def configure_loss(self):
        # Loss function
        self.loss = losses.PoissonNLLDatFilterLoss(self.hparams)
        self.val_loss = self.loss

    def configure_optimizers(self, opt_params=None):
        # Assign optimizer params
        if opt_params is None:
            from neureye.models.utils import create_optimizer_params
            opt_params = create_optimizer_params()
        
        # Assign optimizer
        if opt_params['optimizer']=='AdamW':

            # weight decay only affects certain parameters
            decay = []
            
            decay_names = []
            no_decay_names = []
            no_decay = []
            for name, m in self.named_parameters():
                print('checking {}'.format(name))
                if 'weight' in name:
                    decay.append(m)
                    decay_names.append(name)
                else:
                    no_decay.append(m)
                    no_decay_names.append(name)

            optimizer = torch.optim.AdamW([{'params': no_decay, 'weight_decay': 0}, {'params': decay, 'weight_decay': opt_params['weight_decay']}],
                    lr=opt_params['learning_rate'],
                    betas=opt_params['betas'],
                    amsgrad=opt_params['amsgrad'])

        elif opt_params['optimizer']=='Adam':
            optimizer = torch.optim.Adam(self.parameters(),
                    lr=opt_params['learning_rate'],
                    betas=opt_params['betas'])

        elif opt_params['optimizer']=='LBFGS':
            if 'max_iter' in opt_params:
                max_iter = opt_params['max_iter']
            else:
                max_iter = 4
            if 'history_size' in opt_params:
                history_size = opt_params['history_size']
            else:
                history_size = 10

            optimizer = torch.optim.LBFGS(self.parameters(), history_size=history_size,
                            max_iter=max_iter,
                            line_search_fn="strong_wolfe")

        else:
            raise ValueError('optimizer [%s] not supported' %opt_params['optimizer'])
        
        return optimizer
        

    def compute_network_outputs(self, Xs):
        """
        This applies the forwards of each network in sequential order.

        The tricky thing is concatenating multiple-input dimensions together correctly.
        Note that the external inputs is actually in principle a list of inputs
        
        Note this could return net_ins and net_outs, but currently just saving net_outs (no reason for net_ins yet
        
        """
        assert 'networks' in self.__dict__.keys(), "No networks defined in this NDN"

        net_ins, net_outs = [], []
        for ii in range(len(self.networks)):
            if self.networks[ii].ffnets_in is None:
                # then getting external input
                #net_ins.append( [Xs[self.networks[ii].xstim_n]] )
                net_outs.append( self.networks[ii]( [Xs[self.networks[ii].xstim_n]] ) )
            else:
                in_nets = self.networks[ii].ffnets_in
                # Assemble network inputs in list, which will be used by FFnetwork
                inputs = []
                for mm in range(len(in_nets)):
                    inputs.append( net_outs[in_nets[mm]] )

                # This would automatically  concatenate, which will be FFnetwork-specfic instead (and handled in FFnetwork)
                #input_cat = net_outs[in_nets[0]]
                #for mm in range(1, len(in_nets)):
                #    input_cat = torch.cat( (input_cat, net_outs[in_nets[mm]]), 1 )

                #net_ins.append( inputs )
                net_outs.append( self.networks[ii](inputs) ) 
        return net_ins, net_outs
    # END compute_network_outputs

    def forward(self, Xs):
        
        net_ins, net_outs = self.compute_network_outputs( Xs )
        # For now assume its just one output, given by the first value of self.ffnet_out
        return net_outs[self.ffnet_out[0]]
    # END Encoder.forward

    def training_step(self, batch, batch_idx=None):  # batch_idx is from lightning
        
        y = batch['robs']
        dfs = batch['dfs']

        y_hat = self(batch)

        loss = self.loss(y_hat, y, dfs)

        regularizers = self.compute_reg_loss()

        return {'loss': loss + regularizers, 'train_loss': loss, 'reg_loss': regularizers}
    # END Encoder.training_step

    def validation_step(self, batch, batch_idx=None):
        
        y = batch['robs']
        dfs = batch['dfs']

        y_hat = self(batch)

        loss = self.val_loss(y_hat, y, dfs)
        
        reg_loss = self.compute_reg_loss()
        
        return {'loss': loss, 'val_loss': loss, 'reg_loss': reg_loss}
    
    def compute_reg_loss(self):
        
        rloss = 0
        for network in self.networks:
            rloss += network.compute_reg_loss()
        return rloss
    
    def get_trainer(self, dataset,
        version=None,
        save_dir='./checkpoints',
        name='jnkname',
        optimizer = None,
        scheduler = None,
        opt_params = None):
        """
            Returns a trainer and object splits the training set into "train" and "valid"
        """
        from trainers import Trainer, EarlyStopping
        import os
    
        model = self

        if optimizer is None:
            optimizer = self.get_optimizer(opt_params)

        if opt_params['early_stopping']:
            if isinstance(opt_params['early_stopping'], EarlyStopping):
                earlystopping = opt_params['early_stopping']
            elif isinstance(opt_params['early_stopping'], dict):
                earlystopping = EarlyStopping(patience=opt_params['early_stopping']['patience'],delta=opt_params['early_stopping']['delta'])
            else:
                earlystopping = EarlyStopping(patience=opt_params['early_stopping_patience'],delta=0.0)
        else:
            earlystopping = None

        # Check for device assignment in opt_params
        if opt_params['device'] is not None:
            device = torch.device(opt_params['device'])
        else:
            device = None

        if 'optimize_graph' in opt_params.keys(): # specifies whether to attempt to optimize the graph
            optimize_graph = opt_params['optimize_graph']
            # NOTE: will cause big problems if the batch size is variable
        else:
            optimize_graph = False

        trainer = Trainer(model, optimizer, early_stopping=earlystopping,
                dirpath=os.path.join(save_dir, name),
                optimize_graph=optimize_graph,
                device=device,
                scheduler=scheduler,
                version=version) # TODO: how do we want to handle name? Variable name is currently unused

        return trainer

    def get_dataloaders(self, dataset, batch_size=10, num_workers=4, train_inds=None, val_inds=None, data_seed=None):
        from torch.utils.data import DataLoader, random_split

        if train_inds is None or val_inds is None:
            # check dataset itself
            if hasattr(dataset, 'val_inds') and \
                (dataset.train_inds is not None) and (dataset.val_inds is not None):
                train_inds = dataset.train_inds
                val_inds = dataset.val_inds
            else:
                n_val = len(dataset)//5
                n_train = (len(dataset)-n_val).astype(int)

                if data_seed is None:
                    train_ds, val_ds = random_split(dataset, lengths=[n_train, n_val], generator=torch.Generator()) # .manual_seed(42)
                else:
                    train_ds, val_ds = random_split(dataset, lengths=[n_train, n_val], generator=torch.Generator().manual_seed(data_seed))
            
                if train_inds is None:
                    train_inds = train_ds.indices
                if val_inds is None:
                    val_inds = val_ds.indices
            
        # build dataloaders:

        # we use a batch sampler to sample the data because it generates indices for the whole batch at one time
        # instead of iterating over each sample. This is both faster (probably) for our cases, and it allows us
        # to use the "Fixation" datasets and concatenate along a variable-length batch dimension
        train_sampler = torch.utils.data.sampler.BatchSampler(
            torch.utils.data.sampler.SubsetRandomSampler(train_inds),
            batch_size=batch_size,
            drop_last=False)
        
        val_sampler = torch.utils.data.sampler.BatchSampler(
            torch.utils.data.sampler.SubsetRandomSampler(val_inds),
            batch_size=batch_size,
            drop_last=False)

        train_dl = DataLoader(dataset, sampler=train_sampler, batch_size=None, num_workers=num_workers)
        valid_dl = DataLoader(dataset, sampler=val_sampler, batch_size=None, num_workers=num_workers)
        return train_dl, valid_dl
    # END NDN.get_dataloaders

    def get_optimizer(self, opt_params):

        # get optimizer: In theory this probably shouldn't happen here because it needs to know the model
        # but this was the easiest insertion point I could find for now
        if opt_params['optimizer']=='AdamW':

            # weight decay only affects certain parameters
            decay = []
            
            decay_names = []
            no_decay_names = []
            no_decay = []
            for name, m in self.named_parameters():
                print('checking {}'.format(name))
                if 'weight' in name:
                    decay.append(m)
                    decay_names.append(name)
                else:
                    no_decay.append(m)
                    no_decay_names.append(name)

            # opt_params['weight_decay']
            optimizer = torch.optim.AdamW([{'params': no_decay, 'weight_decay': 0}, {'params': decay, 'weight_decay': opt_params['weight_decay']}],
                    lr=opt_params['learning_rate'],
                    betas=opt_params['betas'],
                    amsgrad=opt_params['amsgrad'])

            # optimizer = torch.optim.AdamW(self.parameters(),
            #         lr=opt_params['learning_rate'],
            #         betas=opt_params['betas'],
            #         weight_decay=opt_params['weight_decay'],
            #         amsgrad=opt_params['amsgrad'])

        elif opt_params['optimizer']=='Adam':
            optimizer = torch.optim.Adam(self.parameters(),
                    lr=opt_params['learning_rate'],
                    betas=opt_params['betas'])

        elif opt_params['optimizer']=='LBFGS':
            if 'max_iter' in opt_params:
                max_iter = opt_params['max_iter']
            else:
                max_iter = 4
            if 'history_size' in opt_params:
                history_size = opt_params['history_size']
            else:
                history_size = 10

            optimizer = torch.optim.LBFGS(self.parameters(), history_size=history_size,
                            max_iter=max_iter,
                            line_search_fn="strong_wolfe")

        else:
            raise ValueError('optimizer [%s] not supported' %opt_params['optimizer'])
        
        return optimizer
    # END NDN.get_optimizer
    
    def fit(
        self, dataset,
        version=None,
        save_dir=None,
        name=None,
        optimizer=None,
        scheduler=None, 
        train_inds=None,
        val_inds=None,
        seed=None):

        '''
        This is the main training loop.
        Steps:
            1. Get a trainer and dataloaders
            2. Prepare regularizers
            3. Run the main fit loop from the trainer, checkpoint, and save model
        '''
        import time

        if save_dir is None:
            save_dir = self.data_dir
        
        if name is None:
            name = self.model_name

        # Precalculate any normalization needed from the data
        if self.loss_module.unit_normalization: # where should unit normalization go?
            # compute firing rates given dataset
            avRs = self.compute_average_responses(dataset) # use whole dataset seems best versus any specific inds
            self.loss_module.set_unit_normalization(avRs) 

        # Make reg modules
        for network in self.networks:
            network.prepare_regularization()

        # Create dataloaders
        batchsize = self.opt_params['batch_size']
        train_dl, valid_dl = self.get_dataloaders(
            dataset, batch_size=batchsize, train_inds=train_inds, val_inds=val_inds)

        # get trainer 
        trainer = self.get_trainer(
            dataset,
            version=version,
            optimizer=optimizer,
            scheduler=scheduler,
            save_dir=save_dir, name = name,
            opt_params = self.opt_params)

        t0 = time.time()
        trainer.fit(self, train_dl, valid_dl, seed=seed)
        t1 = time.time()

        print('  Fit complete:', t1-t0, 'sec elapsed')
    # END NDN.train
    
    def compute_average_responses( self, dataset, data_inds=None ):
        if data_inds is None:
            data_inds = range(len(dataset))

        #if hasattr( dataset, 'use_units') and (len(dataset.use_units) > 0):
        #    rselect = dataset.use_units
        #else:
        #    rselect = dataset.num
        
        # Iterate through dataset to compute average rates
        Rsum, Tsum = 0, 0
        for tt in data_inds:
            sample = dataset[tt]
            Tsum += torch.sum(sample['dfs'], axis=0)
            Rsum += torch.sum(torch.mul(sample['dfs'], sample['robs']), axis=0)

        return torch.divide( Rsum, Tsum.clamp(1))

    def eval_models(self, data, data_inds=None, bits=False, null_adjusted=True):
        '''
        get null-adjusted log likelihood (if null_adjusted = True)
        bits=True will return in units of bits/spike

        Note that data will be assumed to be a dataset, and data_inds will have to be specified batches
        from dataset.__get_item__()
        '''
        
        # Switch into evalulation mode
        self.eval()

        if isinstance(data, dict): 
            # Then assume that this is just to evaluate a sample: keep original here
            assert data_inds is None, "Cannot use data_inds if passing in a dataset sample."
            m0 = self.cpu()
            yhat = m0(data)
            y = data['robs']
            dfs = data['dfs']

            if self.loss_type == 'poisson':
                #loss = nn.PoissonNLLLoss(log_input=False, reduction='none')
                loss = self.loss_module.lossNR
            else:
                print("This loss-type is not supported for eval_models.")
                loss = None

            LLraw = torch.sum( 
                torch.multiply( 
                    dfs, 
                    loss(yhat, y)),
                    axis=0).detach().cpu().numpy()
            obscnt = torch.sum(
                torch.multiply(dfs, y), axis=0).detach().cpu().numpy()
            
            Ts = np.maximum(torch.sum(dfs, axis=0).detach().cpu().numpy(), 1)

            LLneuron = LLraw / np.maximum(obscnt,1) # note making positive

            if null_adjusted:
                predcnt = torch.sum(
                    torch.multiply(dfs, yhat), axis=0).detach().cpu().numpy()
                rbar = np.divide(predcnt, Ts)
                LLnulls = np.log(rbar)-np.divide(predcnt, np.maximum(obscnt,1))
                LLneuron = -LLneuron - LLnulls             
            return LLneuron  # end of the old method

        else:
            # This will be the 'modern' eval_models using already-defined self.loss_module
            # In this case, assume data is dataset
            if data_inds is None:
                data_inds = np.arange(len(data), dtype='int64')

            LLsum, Tsum, Rsum = 0, 0, 0
            d = next(self.parameters()).device  # device the model is on
            for tt in data_inds:
                data_sample = data[tt]
                for dsub in data_sample.keys():
                    if data_sample[dsub].device != d:
                        data_sample[dsub] = data_sample[dsub].to(d)
                pred = self(data_sample)
                LLsum += self.loss_module.unit_loss( 
                    pred, data_sample['robs'], data_filters=data_sample['dfs'], temporal_normalize=False)
                Tsum += torch.sum(data_sample['dfs'], axis=0)
                Rsum += torch.sum(torch.mul(data_sample['dfs'], data_sample['robs']), axis=0)
            LLneuron = torch.divide(LLsum, Rsum.clamp(1) )

            # Null-adjust
            if null_adjusted:
                rbar = torch.divide(Rsum, Tsum.clamp(1))
                LLnulls = torch.log(rbar)-1
                LLneuron = -LLneuron - LLnulls 
        if bits:
            LLneuron/=np.log(2)

        return LLneuron.detach().cpu().numpy()

    def get_weights(self, ffnet_target=0, layer_target=0, to_reshape=True, time_reverse=False):
        return self.networks[ffnet_target].layers[layer_target].get_weights(to_reshape, time_reverse=time_reverse)

    def get_readout_locations(self):
        """This currently retuns list of readout locations and sigmas -- set in readout network"""
        # Find readout network
        net_n = -1
        for ii in range(len(self.networks)):
            if self.networks[ii].network_type == 'readout':
                net_n = ii
        assert net_n >= 0, 'No readout network found.'
        return self.networks[net_n].get_readout_locations()

    def list_parameters(self, ffnet_target=None, layer_target=None):
        if ffnet_target is None:
            ffnet_target = np.arange(len(self.networks), dtype='int32')
        elif not isinstance(ffnet_target, list):
            ffnet_target = [ffnet_target]
        for ii in ffnet_target:
            assert(ii < len(self.networks)), 'Invalid network %d.'%ii
            print("Network %d:"%ii)
            self.networks[ii].list_parameters(layer_target=layer_target)

    def set_parameters(self, ffnet_target=None, layer_target=None, name=None, val=None ):
        """Set parameters for listed layer or for all layers."""
        if ffnet_target is None:
            ffnet_target = np.arange(len(self.networks), dtype='int32')
        elif not isinstance(ffnet_target, list):
            ffnet_target = [ffnet_target]
        for ii in ffnet_target:
            assert(ii < len(self.networks)), 'Invalid network %d.'%ii
            self.networks[ii].set_parameters(layer_target=layer_target, name=name, val=val)

    def plot_filters(self, cmaps=None, ffnet_target=0, layer_target=0, num_cols=8):
        self.networks[ffnet_target].plot_filters(layer_target=layer_target, cmaps=cmaps, num_cols=num_cols)

    def save_model(self, filename=None, alt_dirname=None):
        """Models will be saved using dill/pickle in the directory above the version
        directories, which happen to be under the model-name itself. This assumes the
        current save-directory (notebook specific) and the model name"""

        import dill
        if alt_dirname is None:
            fn = './checkpoints/'
        else:
            fn = alt_dirname
            if alt_dirname != '/':
                fn += '/'
        if filename is None:
            fn += self.model_name + '.pkl'
        else :
            fn += filename
        print( '  Saving model at', fn)

        with open(fn, 'wb') as f:
            dill.dump(self, f)

    def get_null_adjusted_ll(self, sample, bits=False):
        '''
        get null-adjusted log likelihood
        bits=True will return in units of bits/spike
        '''
        m0 = self.cpu()
        if self.loss_type == 'poisson':
            #loss = nn.PoissonNLLLoss(log_input=False, reduction='none')
            loss = self.loss_module.lossNR
        else:
            print('Whatever loss function you want is not yet here.')
        
        lnull = -loss(torch.ones(sample['robs'].shape)*sample['robs'].mean(axis=0), sample['robs']).detach().cpu().numpy().sum(axis=0)
        #yhat = m0(sample['stim'], shifter=sample['eyepos'])
        yhat = m0(sample)
        llneuron = -loss(yhat,sample['robs']).detach().cpu().numpy().sum(axis=0)
        rbar = sample['robs'].sum(axis=0).numpy()
        ll = (llneuron - lnull)/rbar
        if bits:
            ll/=np.log(2)
        return ll
    
    def get_activations(self, sample, ffnet_target=0, layer_target=0, NL=False):
        """
        Returns the inputs and outputs of a specified ffnet and layer
        Args:
            sample: dictionary of sample data from a dataset
            ffnet_target: which network to target (default: 0)
            layer_target: which layer to target (default: 0)
            NL: get activations using the nonlinearity as the module target
        Output:
            activations: dictionary of activations
            with keys:
                'input' : input to layer
                'output' : output of layer
        """
        
        activations = {}

        def hook_fn(m, i, o):
            activations['input'] = i
            activations['output'] = o

        if NL:
            if self.networks[ffnet_target].layers[layer_target].NL:
                handle = self.networks[ffnet_target].layers[layer_target].NL.register_forward_hook(hook_fn)
            else:
                raise ValueError('This layer does not have a non-linearity. Call with NL=False')
        else:
            handle = self.networks[ffnet_target].layers[layer_target].register_forward_hook(hook_fn)

        out = self(sample)
        handle.remove()
        return activations

    @classmethod
    def load_model(cls, checkpoint_path=None, model_name=None, version=None):
        '''
            Load a model from disk.
            Arguments:
                checkpoint_path: path to directory containing model checkpoints
                model_name: name of model (from model.model_name)
                version: which checkpoint to load (default: best)
            Returns:
                model: loaded model
        '''
        
        from NDNutils import get_fit_versions

        assert checkpoint_path is not None, "Need to provide a checkpoint_path"
        assert model_name is not None, "Need to provide a model_name"

        out = get_fit_versions(checkpoint_path, model_name)
        if version is None:
            version = out['version_num'][np.argmax(np.asarray(out['val_loss']))]
            print("No version requested. Using (best) version (v=%d)" %version)

        assert version in out['version_num'], "Version %d not found in %s. Must be: %s" %(version, checkpoint_path, str(out['version_num']))
        ver_ix = np.where(version==np.asarray(out['version_num']))[0][0]
        # Load the model
        model = torch.load(out['model_file'][ver_ix])
        
        return model

        # import os
        # import dill
        # if alt_dirname is None:
        #     fn = './checkpoints/'
        # else:
        #     fn = alt_dirname
        #     if alt_dirname != '/':
        #         fn += '/'
        # if filename is None:
        #     assert model_name is not None, 'Need model_name or filename.'
        #     fn += model_name + '.pkl'
        # else :
        #     fn += filename

        # if not os.path.isfile(fn):
        #     raise ValueError(str('%s is not a valid filename' %fn))

        # print( 'Loading model:', fn)
        # with open(fn, 'rb') as f:
        #     model = dill.load(f)
        # model.encoder = None
        # if version is not None:
        #     from pathlib import Path
        #     assert filename is None, 'Must recover version from checkpoint dir.'
        #     # Then load checkpointed encoder on top of model
        #     chkpntdir = fn[:-4] + '/version_' + str(version) + '/'
        #     chkpath = Path(chkpntdir) / 'checkpoints'
        #     ckpt_files = list(chkpath.glob('*.ckpt'))
        #     model.encoder = Encoder.load_from_checkpoint(str(ckpt_files[0]))
        #     nn.utils.remove_weight_norm(model.encoder.core.features.layer0.conv)
        #     print( '-> Updated with', str(ckpt_files[0]))

        # return model
