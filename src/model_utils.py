from copy import deepcopy
import numpy as np

def weight_normer(W1):
    '''
    This function takes in a model and normalizes
    its weights, i.e., the norm of the individual
    weight vector are set to one explicitly.
    '''
    '''
    This model has three layers, thus three weight
    matrices. We need to find the norms for each 
    one of these.
    '''
    W1_norms = np.sum(np.abs(W1)**2,axis=-0)**(1./2)
    W1 /= W1_norms
    return W1
