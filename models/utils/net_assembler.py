import torch.nn as nn
import neureye.models.layers.networks as networks

FFnets = {
    'normal': networks.FFnetwork,
    'readout': networks.ReadoutNetwork
}

def assemble_ffnetworks(ffnet_list, external_nets=None):
    """
    This function takes a list of ffnetworks and puts them together 
    in order. This has to do two steps for each ffnetwork: 

    1. Plug in the inputs to each ffnetwork as specified
    2. Builds the ff-network with the input
    
    This returns the a 'network', which is (currently) a module with a 
    'forward' and 'reg_loss' function specified.

    When multiple ffnet inputs are concatenated, it will always happen in the first
    (filter) dimension, so all other dimensions must match
    """
    assert type(ffnet_list) is list, "ffnet_list must be a list."
    
    num_networks = len(ffnet_list)
    # Make list of pytorch modules
    networks = nn.ModuleList()

    for mm in range(num_networks):

        # Determine internal network input to each subsequent network (if exists)
        if ffnet_list[mm]['ffnet_n'] is not None:
            nets_in = ffnet_list[mm]['ffnet_n']
            input_dims_list = []
            for ii in range(len(nets_in)):
                assert nets_in[ii] < mm, "FFnet%d (%d): input networks must come earlier"%(mm, ii)
                input_dims_list.append(networks[nets_in[ii]].layers[-1].output_dims)
            ffnet_list[mm]['input_dims_list'] = input_dims_list
    
        # Create corresponding FFnetwork
        net_type = ffnet_list[mm]['ffnet_type']
        if net_type == 'external':  # separate case because needs to pass in external modules directly
            networks.append( networks.FFnet_external(ffnet_list[mm], external_nets))
        else:
            networks.append( FFnets[net_type](**ffnet_list[mm]) )

    return networks