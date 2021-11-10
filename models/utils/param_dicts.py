from copy import deepcopy

#################### CREATE NETWORK PARAMETER-DICTS ####################
def layer_dict(
    input_dims=None, num_filters=1, NLtype='relu', 
    norm_type=0, pos_constraint=False, num_inh=0,
    conv=False, conv_width=None, stride=None, dilation=None):

    """input dims are [num_filters, space1, space2, num_lags]"""

    # Add any other nonlinaerities here (and pass to functionals below)
    val_nls = ['lin', 'relu', 'elu', 'quad', 'softplus', 'tanh', 'sigmoid']
    assert NLtype in val_nls, 'NLtype not valid.'

    output_dims = [num_filters, 1, 1, 1]
    if conv:
        output_dims[1:3] = input_dims[1:3]
        if conv_width is None:
            TypeError( 'Need to define conv filter-width.')
        filter_dims = [input_dims[0], conv_width, conv_width, input_dims[3]]
        if input_dims[2] == 1:  # then 1-d spatial
            filter_dims[2] = 1
    else:
        filter_dims = deepcopy(input_dims)

    if num_inh > num_filters:
        print("Warning: num_inh is too large. Adjusted to ", num_filters)
        num_inh = num_filters
        
    if conv:
        if stride is None:
            stride = 1
        if dilation is None:
            dilation = 1
            
    params_dict = {
        'input_dims': input_dims,
        'output_dims': output_dims,
        'num_filters': num_filters,
        'filter_dims': filter_dims, 
        'NLtype': NLtype, 
        'norm_type': norm_type, 
        'pos_constraint': pos_constraint,
        'num_inh': num_inh,
        'conv': conv,
        'stride': stride,
        'dilation': dilation,
        'temporal_tent_spacing': None,  # only useful for STconv thus far
        'weights_initializer': 'uniform',
        'bias_initializer': 'zeros',
        'bias': True}

    return params_dict
# END layer_dict

def ffnet_params_default(xstim_n=None, ffnet_n=None, input_dims=None):
    """This creates a ffnetwork_params object that specifies details of ffnetwork
    ffnetwork dicts have these fields:
        ffnet_type: string specifying network type (default = 'normal')
        layer_list: defaults to None, to be set in internal function
        input_dims: defaults to None, but will be set when network made, if not sooner
        xstim_n: external input (or None). Note can only be from one source
        ffnet_n: list of internal network inputs (has to be a list, or None)
        conv: [boolean] whether ffnetwork is convolutional or not, defaults to False but to be set
        
    -- Note xstim_n and ffnet_n are created/formatted here. 
    -- If xstim_n is specified, it must specify input dimensions
    -- This should set all required fields as needed (even if none)"""

    if ffnet_n is None:
        if xstim_n is None:
            xstim_n = 'stim'  # default to first external input if unspecified
    else:
        # currently does not concatenate internal and external inputs (should it?)
        assert xstim_n is None, "Currently cannot have external and internal inputs."
        if type(ffnet_n) is not list:
            ffnet_n = [ffnet_n]

    ffnet_params = {
        'ffnet_type': 'normal',
        'layer_types': None,
        'layer_list': None,
        'input_dims_list': [input_dims],
        'xstim_n': xstim_n,
        'ffnet_n': ffnet_n,
        'conv': False,
        'reg_list': None
        }
    return ffnet_params
# END ffnet_params_default


def ffnet_dict_NIM(
    input_dims=None, 
    layer_sizes=None, 
    layer_types=None,
    act_funcs=None,
    ei_layers=None,
    conv_widths=None,
    norm_list=None,
    reg_list=None,
    xstim_n='stim',
    ffnet_n = None,
    ffnet_type='normal'):

    """This creates will make a list of layer dicts corresponding to a non-convolutional NIM].
    Note that input_dims can be set to none"""

    ffnet_params = ffnet_params_default(xstim_n=xstim_n, ffnet_n=ffnet_n)

    assert ffnet_type in ['normal', 'add', 'mult'], "ffnet_type must be 'normal', 'add', or 'mult' for this type of network."
    ffnet_params['ffnet_type'] = ffnet_type
    ffnet_params['input_dims_list'] = [input_dims]
    ffnet_params['reg_list'] = reg_list

    num_layers = len(layer_sizes)
    assert len(act_funcs) == num_layers, "act_funcs is wrong length."

    indims = input_dims  # starts with network input dims

    ei_layers = list_complete(ei_layers, L=num_layers, null_val=0)
    norm_list = list_complete(norm_list, L=num_layers, null_val=0)
    layer_types = list_complete(layer_types, L=num_layers, null_val='normal')
    ffnet_params['layer_types'] = layer_types

    conv_widths = list_complete(conv_widths, L=num_layers)

    layer_list = []   
    for ll in range(num_layers):
        pos_con = False
        if ll > 0:
            if ei_layers[ll-1] is not None:
                pos_con = True
        if ei_layers[ll] is None:
            num_inh = 0
        else:
            num_inh = ei_layers[ll]

        layer_list.append(
            layer_dict(
                input_dims = indims, 
                num_filters = layer_sizes[ll],
                NLtype = act_funcs[ll],
                norm_type = norm_list[ll],
                pos_constraint = pos_con,
                conv="conv" in layer_types[ll],
                conv_width = conv_widths[ll], 
                num_inh = num_inh))

        indims = layer_list[-1]['output_dims']
    ffnet_params['layer_list'] = layer_list

    return ffnet_params
# END ffnet_dict_NIM


def ffnet_dict_readout(
    ffnet_n=None,
    num_cells=0,
    act_func='softplus',
    bias=True,
    init_mu_range=0.1,
    init_sigma=1,
    batch_sample=True,
    align_corners=True,
    gauss_type='uncorrelated',
    pos_constraint=False,
    reg_list=None):
    """This sets up dictionary parameters for readout ffnetwork, establishing all the relevant info. Note that the shifter
    is designated as the second of two ffnet_n inputs listed, so is not separately specified."""

    assert ffnet_n is not None, 'Must specify input ffnetwork (ffnet_n).'
    assert num_cells > 0, 'Must specify num_cells.'

    ffnet_params = ffnet_params_default(xstim_n=None, ffnet_n=ffnet_n)
    ffnet_params['ffnet_type'] = 'readout'
    ffnet_params['layer_types'] = ['readout']
    
    # save rest of params in ffnet dictionary
    ffnet_params['reg_list'] = reg_list   
    layer_params = layer_dict(
        input_dims = None, 
        num_filters = num_cells,
        NLtype = act_func,
        pos_constraint = pos_constraint)

    # Add extra parameters to get passed into layer function
    layer_params['bias'] = bias
    layer_params['init_mu_range'] = init_mu_range
    layer_params['init_sigma'] = init_sigma
    layer_params['batch_sample'] = batch_sample
    layer_params['align_corners'] = align_corners
    layer_params['gauss_type'] = gauss_type

    ffnet_params['layer_list'] = [layer_params]

    return ffnet_params


def ffnet_dict_external(
    name='external',
    xstim_n='stim',
    ffnet_n=None,
    input_dims=None, 
    input_dims_reshape=None,
    output_dims=None):
    """The network information passed in must simply be:
        1. The name of the network in the network dictionary that is passed into the constructor.
        2. The source of network input (external stim or otherwise).
        3. If external input, its input dims must also be specified.
        4. If the network takes input that needs to be reshaped, pass in 'input_dims_reshaped' that adds the 
        batch dimension and then reshapes before passing into external network. Note this will be the dimensionality
        that the network takes as input, so does not need to be the NDN convention [CWHT]
        5. The output dims that are passed to the rest of the network (needs to be [CWHT]"""

    ffnet_params = ffnet_params_default(xstim_n=xstim_n, ffnet_n=ffnet_n, input_dims=input_dims)
    ffnet_params['ffnet_type'] = 'external'
    ffnet_params['layer_types'] = ['external']
    ffnet_params['input_dims_reshape'] = input_dims_reshape
    ffnet_params['external_module_name'] = name
    
    if not isinstance(output_dims, list):  # then passing out just filters?
        output_dims = [output_dims, 1, 1, 1]

    layer_params = layer_dict(
        input_dims = input_dims, 
        num_filters = output_dims[0],  # first argument of passed-in output_dims, by definition
        NLtype = 'lin') # this is dummy value so doesnt hurt calling this dict)

    layer_params['output_dims'] = output_dims
    layer_params['bias'] = False

    ffnet_params['layer_list'] = [layer_params]

    return ffnet_params

def list_complete( fauxlist, L=None, null_val=None ):
    # Makes 'valid' list from a faux_list that might be too short or not a list at all

    if fauxlist is None:
        assert L is not None, 'Internal: Requires length specification'
        return [null_val]*L
    if not isinstance( fauxlist, list):
        fauxlist = [fauxlist]
    
    clist = deepcopy(fauxlist)
    if L is not None:
        while len(clist) < L:
            clist.append(null_val)
    return clist

def create_optimizer_params(
        batch_size=1000,
        weight_decay=None,
        early_stopping=True,
        early_stopping_patience=4,
        max_iter=10000,
        learning_rate=1e-3,
        betas=[0.9, 0.999],
        num_gpus=1,
        progress_bar_refresh=20, # num of batches 
        num_workers=4,
        line_search='Wolfe', # see LBFGS for meaning of line_search method
        history=10):

    if early_stopping:
        max_epochs = 1000
    else:
        max_epochs = 300

    optpar = {
        'batch_size': batch_size,
        'device': None,
        'weight_decay': weight_decay,
        'early_stopping': early_stopping,
        'early_stopping_patience': early_stopping_patience,
        'max_iter': max_iter,
        'max_epochs': max_epochs,
        'learning_rate': learning_rate,
        'betas': betas, 
        'amsgrad': False,
        'auto_lr': False,
        'progress_bar_refresh': progress_bar_refresh,
        'num_workers': num_workers,
        'num_gpus': num_gpus,
        'history': history,
        'line_search': line_search,
        'optimizer': 'AdamW'}

    return optpar