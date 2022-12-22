import os
import shutil
import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy
import inspect


class hyperparams(dict):
    def __init__(self, *args, **kwargs):
        super(hyperparams, self).__init__(*args, **kwargs)
        self.__dict__ = self

def save_hyperparameters():
    frame = inspect.currentframe().f_back
    _, _, _, local_vars = inspect.getargvalues(frame)
    local_vars.pop('self')
    
    for key in local_vars.keys():
        if isinstance(local_vars[key], nn.Module):
            local_vars[key] = str(local_vars[key])
    
    hparams = hyperparams(local_vars)
    if '__class__' in hparams.keys():
        hparams.pop('__class__')

    return hparams

def get_fit_versions(save_path):
    '''
        Find versions of the fit model
        Arguments:
            data_dir: directory where the checkpoints are stored
            model_name: name of the model
    '''

    import re
    from tensorboard.backend.event_processing import event_accumulator

    dirlist = [x for x in os.listdir(save_path) if os.path.isdir(os.path.join(save_path, x))]
        
    versionlist = [re.findall('(?!version)\d+', x) for x in dirlist]
    versionlist = [int(x[0]) for x in versionlist if not not x]

    outdict = {'version_num': [],
        'events_file': [],
        'model_file': [],
        'val_loss': [],
        'val_loss_steps': []}

    for v in versionlist:
        # events_file = os.path.join(data_dir, model_name, 'version_%d' %v, 'events.out.tfevents.%d' %v)
        vpath = os.path.join(save_path, 'version%d' %v)
        vplist = os.listdir(vpath)

        tfeventsfiles = [x for x in vplist if 'events.out.tfevents' in x]
        modelfiles = [x for x in vplist if 'model.pt' in x]

        if len(tfeventsfiles) > 0 and len(modelfiles) == 1:
            evfile = os.path.join(vpath, tfeventsfiles[0])
            # read from tensorboard backend
            ea = event_accumulator.EventAccumulator(evfile)
            ea.Reload()
            try:
                val = np.asarray([x.value for x in ea.scalars.Items("Loss/Validation (Epoch)")])
                bestval = np.nanmin(val)

                outdict['version_num'].append(v)
                outdict['events_file'].append(evfile)
                outdict['model_file'].append(os.path.join(vpath, modelfiles[0]))
                outdict['val_loss_steps'].append(val)
                outdict['val_loss'].append(bestval)
            except:
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
    yhat = m0(sample['stim'], shifter=sample['eyepos'], sample=sample)
    llneuron = -loss(yhat,sample['robs']).detach().cpu().numpy().sum(axis=0)
    rbar = sample['robs'].sum(axis=0).numpy()
    ll = (llneuron - lnull)/rbar
    if bits:
        ll/=np.log(2)
    return ll

def save_checkpoint(state, save_path: str, is_best: bool = False, max_keep: int = None):
    """Saves torch model to checkpoint file.
    Args:
        state (torch model state): State of a torch Neural Network (use model.state_dict() to get it)
        save_path (str): Destination path for saving checkpoint
        is_best (bool): If ``True`` creates additional copy
            ``best_model.ckpt``
        max_keep (int): Specifies the max amount of checkpoints to keep
    
    credit: pulled from https://github.com/IgorSusmelj/pytorch-styleguide
    """

    # save checkpoint
    torch.save(state, save_path)

    # deal with max_keep
    save_dir = os.path.dirname(save_path)
    list_path = os.path.join(save_dir, 'latest_checkpoint.txt')

    save_path = os.path.basename(save_path)
    if os.path.exists(list_path):
        with open(list_path) as f:
            ckpt_list = f.readlines()
            ckpt_list = [save_path + '\n'] + ckpt_list
    else:
        ckpt_list = [save_path + '\n']

    if max_keep is not None:
        for ckpt in ckpt_list[max_keep:]:
            ckpt = os.path.join(save_dir, ckpt[:-1])
            if os.path.exists(ckpt):
                os.remove(ckpt)
        ckpt_list[max_keep:] = []

    with open(list_path, 'w') as f:
        f.writelines(ckpt_list)

    # copy best
    if is_best:
        shutil.copyfile(save_path, os.path.join(save_dir, 'best_model.ckpt'))


def load_checkpoint(ckpt_dir_or_file: str, map_location=None, load_best=False):
    """Loads torch model from checkpoint file.
    Args:
        ckpt_dir_or_file (str): Path to checkpoint directory or filename
        map_location: Can be used to directly load to specific device
        load_best (bool): If True loads ``best_model.ckpt`` if exists.

    credit: pulled from https://github.com/IgorSusmelj/pytorch-styleguide
    """
    if os.path.isdir(ckpt_dir_or_file):
        if load_best:
            ckpt_path = os.path.join(ckpt_dir_or_file, 'best_model.ckpt')
        else:
            with open(os.path.join(ckpt_dir_or_file, 'latest_checkpoint.txt')) as f:
                ckpt_path = os.path.join(ckpt_dir_or_file, f.readline()[:-1])
    else:
        ckpt_path = ckpt_dir_or_file
    ckpt = torch.load(ckpt_path, map_location=map_location)
    print(' [*] Loading checkpoint from %s succeed!' % ckpt_path)
    return ckpt


def ensure_dir(dir_name: str):
    """Creates folder if not exists.

    credit: pulled from https://github.com/IgorSusmelj/pytorch-styleguide
    """
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)



def ModelSummary(model, input_size, batch_size=-1, device=torch.device('cuda:0'), dtypes=None):
    result, params_info = summary_string(
        model, input_size, batch_size, device, dtypes)
    print(result)

    return params_info


def summary_string(model, input_size, batch_size=-1, device=torch.device('cuda:0'), dtypes=None):
    if dtypes == None:
        dtypes = [torch.FloatTensor]*len(input_size)

    summary_str = ''

    def register_hook(module):
        def hook(module, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)

            m_key = "%s-%i" % (class_name, module_idx + 1)
            summary[m_key] = OrderedDict()
            summary[m_key]["input_shape"] = list(input[0][0].size())
            #summary[m_key]["input_shape"] = list(input[0].size())   # CHANGED to above: inputs are now list 
            summary[m_key]["input_shape"][0] = batch_size
            if isinstance(output, (list, tuple)):
                summary[m_key]["output_shape"] = [
                    [-1] + list(o.size())[1:] for o in output
                ]
            else:
                summary[m_key]["output_shape"] = list(output.size())
                summary[m_key]["output_shape"][0] = batch_size

            params = 0
            if hasattr(module, "weight") and hasattr(module.weight, "size"):
                params += torch.prod(torch.LongTensor(list(module.weight.size())))
                summary[m_key]["trainable"] = module.weight.requires_grad
            if hasattr(module, "bias") and hasattr(module.bias, "size"):
                params += torch.prod(torch.LongTensor(list(module.bias.size())))
            summary[m_key]["nb_params"] = params

        if (
            not isinstance(module, nn.Sequential)
            and not isinstance(module, nn.ModuleList)
        ):
            hooks.append(module.register_forward_hook(hook))

    # multiple inputs to the network
    if isinstance(input_size, tuple):
        input_size = [input_size]

    # batch_size of 2 for batchnorm
    x = [torch.rand(2, *in_size).type(dtype).to(device=device)
         for in_size, dtype in zip(input_size, dtypes)]

    # create properties
    summary = OrderedDict()
    hooks = []

    # register hook
    model.apply(register_hook)

    # make a forward pass
    # print(x.shape)
    model(*x)

    # remove these hooks
    for h in hooks:
        h.remove()

    summary_str += "----------------------------------------------------------------" + "\n"
    line_new = "{:>20}  {:>25} {:>15}".format(
        "Layer (type)", "Output Shape", "Param #")
    summary_str += line_new + "\n"
    summary_str += "================================================================" + "\n"
    total_params = 0
    total_output = 0
    trainable_params = 0
    for layer in summary:
        # input_shape, output_shape, trainable, nb_params
        line_new = "{:>20}  {:>25} {:>15}".format(
            layer,
            str(summary[layer]["output_shape"]),
            "{0:,}".format(summary[layer]["nb_params"]),
        )
        total_params += summary[layer]["nb_params"]

        total_output += np.prod(summary[layer]["output_shape"])
        if "trainable" in summary[layer]:
            if summary[layer]["trainable"] == True:
                trainable_params += summary[layer]["nb_params"]
        summary_str += line_new + "\n"

    # assume 4 bytes/number (float on cuda).
    total_input_size = abs(np.prod(sum(input_size, ()))
                           * batch_size * 4. / (1024 ** 2.))
    total_output_size = abs(2. * total_output * 4. /
                            (1024 ** 2.))  # x2 for gradients
    total_params_size = abs(total_params * 4. / (1024 ** 2.))
    total_size = total_params_size + total_output_size + total_input_size

    summary_str += "================================================================" + "\n"
    summary_str += "Total params: {0:,}".format(total_params) + "\n"
    summary_str += "Trainable params: {0:,}".format(trainable_params) + "\n"
    summary_str += "Non-trainable params: {0:,}".format(total_params -
                                                        trainable_params) + "\n"
    summary_str += "----------------------------------------------------------------" + "\n"
    summary_str += "Input size (MB): %0.2f" % total_input_size + "\n"
    summary_str += "Forward/backward pass size (MB): %0.2f" % total_output_size + "\n"
    summary_str += "Params size (MB): %0.2f" % total_params_size + "\n"
    summary_str += "Estimated Total Size (MB): %0.2f" % total_size + "\n"
    summary_str += "----------------------------------------------------------------" + "\n"
    # return summary
    return summary_str, (total_params, trainable_params)