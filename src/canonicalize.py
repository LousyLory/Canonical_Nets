from copy import deepcopy
import numpy as np

def canon_nets(model2, indices_to_copy_to, use_batchnorm = False):
    '''
    This function creates canonicalized networks
    '''
    model2p = deepcopy(model2)

    # altering networks
    model2p.params['W1'] = model2p.params['W1'][:,indices_to_copy_to]
    model2p.params['b1'] = model2p.params['b1'][indices_to_copy_to]
    if use_batchnorm == True:
        model2p.params['gamma0'] = model2p.params['gamma0'][indices_to_copy_to]
        model2p.params['beta0'] = model2p.params['beta0'][indices_to_copy_to]
        model2p.bn_params[0]['running_mean'] = model2p.bn_params[0]['running_mean'][indices_to_copy_to]
        model2p.bn_params[0]['running_var'] = model2p.bn_params[0]['running_var'][indices_to_copy_to]
    
    model2p.params['W2'] = model2p.params['W2'][indices_to_copy_to, :]
    model2p.params['b2'] = model2p.params['b2'][indices_to_copy_to]
    if use_batchnorm == True:
        model2p.params['gamma1'] = model2p.params['gamma1'][indices_to_copy_to]
        model2p.params['beta1'] = model2p.params['beta1'][indices_to_copy_to]
        model2p.bn_params[1]['running_mean'] = model2p.bn_params[1]['running_mean'][indices_to_copy_to]
        model2p.bn_params[1]['running_var'] = model2p.bn_params[1]['running_var'][indices_to_copy_to]
    
    model2p.params['W3'] = np.copy(model2.params['W3'])
    model2p.params['b3'] = np.copy(model2.params['b3'])
    
    return model2p
