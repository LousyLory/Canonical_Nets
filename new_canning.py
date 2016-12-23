# As usual, a bit of setup

import time
import numpy as np
import matplotlib.pyplot as plt
from src.classifiers.fc_net import *
from src.data_utils import get_CIFAR10_data
from src.gradient_check import eval_numerical_gradient, eval_numerical_gradient_array
from src.solver import Solver
from src.canonicalize import *

def rel_error(x, y):
  """ returns relative error """
  return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

# Load the (preprocessed) CIFAR10 data.

data = get_CIFAR10_data()
for k, v in data.iteritems():
  print '%s: ' % k, v.shape

# splitting up the train data

X_train = data['X_train']
y_train = data['y_train']

total_train_size = X_train.shape[0]
full_indices = np.random.permutation(total_train_size)

mask1 = range(total_train_size/3)
X_train1 = X_train[full_indices[mask1]]
y_train1 = y_train[full_indices[mask1]]

mask2 = range(total_train_size/3, 2*total_train_size/3)
X_train2 = X_train[full_indices[mask2]]
y_train2 = y_train[full_indices[mask2]]

mask3 = range(2*total_train_size/3, 3*total_train_size/3)
X_train3 = X_train[full_indices[mask3]]
y_train3 = y_train[full_indices[mask3]]

data1 = {
        'X_train': X_train1, 'y_train' : y_train1,
        'X_val': data['X_val'], 'y_val' : data['y_val'],
        'X_test': data['X_test'], 'y_test' : data['y_test'],
        }
data2 = {
        'X_train': X_train2, 'y_train' : y_train2,
        'X_val': data['X_val'], 'y_val' : data['y_val'],
        'X_test': data['X_test'], 'y_test' : data['y_test'],
        }
data3 = {
        'X_train': X_train3, 'y_train' : y_train3,
        'X_val': data['X_val'], 'y_val' : data['y_val'],
        'X_test': data['X_test'], 'y_test' : data['y_test'],
        }

# train models
from src.creator import *
from src.canonicalize import *
from src.required_funcs import *

accuracy = np.zeros((1,5))
alpha = 0.5

for net in range(1):
    # counter printer
    print "iteration: ", net
    
    # train net1
    model1 = FullyConnectedNet([100, 100], weight_scale=0.003, use_batchnorm = False, reg=0.6)
    solver1 = Solver(model1, data1,
                            print_every=data1['X_train'].shape[0], num_epochs=50, batch_size=100,
                            update_rule='sgd',
                            optim_config={
                              'learning_rate': 0.03,
                            },verbose=False, lr_decay = 0.9,
                     )
    solver1.train()
    pass
    #train net2
    model2 = FullyConnectedNet([100, 100], weight_scale=0.003, use_batchnorm = False, reg=0.6)
    solver2 = Solver(model2, data1,
                            print_every=data1['X_train'].shape[0], num_epochs=50, batch_size=100,
                            update_rule='sgd',
                            optim_config={
                              'learning_rate': 0.03,
                            },verbose=False, lr_decay = 0.9,
                     )
    solver2.train()
    pass
    # obtain accuracy of net1
    y_val_pred = np.argmax(model1.loss(data1['X_val']), axis=1)
    accuracy[net,0] = (y_val_pred == data1['y_val']).mean()
    print accuracy[net,0]
    
    # obtain accuracy of net2
    y_val_pred = np.argmax(model2.loss(data1['X_val']), axis=1)
    accuracy[net,1] = (y_val_pred == data1['y_val']).mean()
    print accuracy[net,1]
    
    # average network generation
    avg_model = create_model(model1, model2, alpha)
    # avgerage network performance
    y_val_pred = np.argmax(avg_model.loss(data1['X_val']), axis=1)
    accuracy[net,2] = (y_val_pred == data1['y_val']).mean()
    print accuracy[net,2]
    
    # canonicalizing model2 to look like model1
    # greedy method
    indices, model1, model2 = match_vals(model1, model2, method='greedy', is_normed='Erik')
    canon_model2_greedy = canon_nets(model2, indices.astype(int), use_batchnorm = False)
    # average network generation
    avg_model_can_greedy = create_model(model1, canon_model2_greedy, alpha)
    # avgerage network performance
    y_val_pred = np.argmax(avg_model_can_greedy.loss(data1['X_val']), axis=1)
    accuracy[net,3] = (y_val_pred == data1['y_val']).mean()
    print accuracy[net,3]
    
    # hungarian method
    indices, model1, model2 = match_vals(model1, model2, method='Hungarian', is_normed='Erik')
    '''
    # checking new models
    y_val_pred = np.argmax(model1.loss(data1['X_val']), axis=1)
    print (y_val_pred == data1['y_val']).mean()
    '''
    canon_model2_hung = canon_nets(model2, indices.astype(int), use_batchnorm = False)
    # average network generation
    avg_model_can_hung = create_model(model1, canon_model2_hung, alpha)
    # avgerage network performance
    y_val_pred = np.argmax(avg_model_can_hung.loss(data1['X_val']), axis=1)
    accuracy[net,4] = (y_val_pred == data1['y_val']).mean()
    print accuracy[net,4]
    
    '''
    # canonicalizing model1 to look like model2
    # greedy method
    indices = match_vals(model2, model1, method='greedy')
    canon_model2_greedy = canon_nets(model1, indices.astype(int), use_batchnorm = False)
    # average network generation
    avg_model_can_greedy = create_model(model2, canon_model2_greedy, alpha)
    # avgerage network performance
    y_val_pred = np.argmax(avg_model_can_greedy.loss(data1['X_val']), axis=1)
    accuracy[net,5] = (y_val_pred == data1['y_val']).mean()
    
    # hungarian method
    indices = match_vals(model2, model1, method='Hungarian')
    canon_model2_hung = canon_nets(model1, indices.astype(int), use_batchnorm = False)
    # average network generation
    avg_model_can_hung = create_model(model2, canon_model2_hung, alpha)
    # avgerage network performance
    y_val_pred = np.argmax(avg_model_can_hung.loss(data1['X_val']), axis=1)
    accuracy[net,6] = (y_val_pred == data1['y_val']).mean()
    '''
    
import numpy, scipy.io

scipy.io.savemat('arrdata_new_canning.mat', mdict={'accuracy': accuracy})