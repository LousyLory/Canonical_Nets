from munkres import Munkres
import numpy as np
from dist_fun import *

def match_vals(model1p, model2p, method = 'Hungarian'):
    '''
    This method defines matching vectors on the basis of function
    '''
    matrix = compute_distances_no_loops(model1p.params['W1'].T, model2p.params['W1'].T)
    try:
        if method == 'Hungarian':
            m = Munkres()
            indices = m.compute(matrix)
            indices_assignments = np.copy(indices)
            indices_to_copy_to = np.copy(indices_assignments[:,1])
            feat = np.copy(indices_to_copy_to)
            
    except ValueError:
        print("run from command line: pip install munkres")
    
    if method == 'greedy':
        indices = np.argsort(matrix, axis=0)
        used = np.array(range(indices.shape[0]))
        feat = np.zeros(indices.shape[0])
        for i in range(indices.shape[0]):
            for j in range(indices.shape[0]):
                if indices[i,j] in used:
                    feat[i] = indices[i,j]
                    used[indices[i,j]] = 100
                    break
    return feat