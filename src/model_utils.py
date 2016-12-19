from copy import deepcopy
import numpy as np

def weight_nomrer(model):
	'''
	This function takes in a model and normalizes
	its weights, i.e., the norm of the individual
	weight vector are set to one explicitly.
	'''
	model1 = deepcopy(model)
	'''
	This model has three layers, thus three weight
	matrices. We need to find the norms for each 
	one of these.
	'''
	W1_norms = np.sum(np.abs(model1.params['W1'])**2,axis=-0)**(1./2)
	W1 = np.copy(model1.params['W1'])
	W1 /= W1_norms
	model1.params['W1'] = np.copy(W1)
	return model1, W1_norms
