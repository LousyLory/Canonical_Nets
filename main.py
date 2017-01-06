import time
import numpy as np
import matplotlib.pyplot as plt
from src.classifiers.fc_net import *
from src.data_utils import get_CIFAR10_data
from src.gradient_check import eval_numerical_gradient, eval_numerical_gradient_array
from src.solver import Solver
from src.canonicalize import *
import pickle

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

data1 = {
        'X_train': X_train1, 'y_train' : y_train1,
        'X_val': data['X_val'], 'y_val' : data['y_val'],
        'X_test': data['X_test'], 'y_test' : data['y_test'],
        }

# external variables
accuracy = np.zeros((200,6))
alpha = 0.5

from src.creator import *
from src.model_utils import *

# main loop
for net in range(200):
    print net
    name = 'models/' + str(net) + '.pkl'
    with open(name, 'rb') as input:
        model1 = pickle.load(input)
        model2 = pickle.load(input)
    
    # step1: normalize models
    
    model1q = model_normer(model1)
    model2q = model_normer(model2)
    
    # step2: canonicalize models
    model2p_hung = full_canon_nets(model1q, model2q, method = 'Hungarian')
    model2p_gred = full_canon_nets(model1q, model2q, method = 'greedy')
    
    # step3: interpolate with fixed alpha
    new_model_original = create_model(model1, model2, alpha)
    new_model_normed = create_model(model1q, model2q, alpha)
    new_model_canoned_hung = create_model(model1q, model2p_hung, alpha)
    new_model_canoned_gred = create_model(model1q, model2p_gred, alpha)
    
    # calculate accuracies on train data
    # original model 1
    y_val_pred = np.argmax(model1q.loss(data1['X_val']), axis=1)
    accuracy[net,0] = (y_val_pred == data1['y_val']).mean()
    # original model 2
    y_val_pred = np.argmax(model2q.loss(data1['X_val']), axis=1)
    accuracy[net,1] = (y_val_pred == data1['y_val']).mean()
    # new model from originals
    y_val_pred = np.argmax(new_model_original.loss(data1['X_val']), axis=1)
    accuracy[net,2] = (y_val_pred == data1['y_val']).mean()
    # new model from normed
    y_val_pred = np.argmax(new_model_normed.loss(data1['X_val']), axis=1)
    accuracy[net,3] = (y_val_pred == data1['y_val']).mean()
    # new model from canoned using hungarian
    y_val_pred = np.argmax(new_model_canoned_hung.loss(data1['X_val']), axis=1)
    accuracy[net,4] = (y_val_pred == data1['y_val']).mean()
    # new model from canoned using greedy
    y_val_pred = np.argmax(new_model_canoned_gred.loss(data1['X_val']), axis=1)
    accuracy[net,5] = (y_val_pred == data1['y_val']).mean()

import numpy, scipy.io

scipy.io.savemat('arrdata_full_canning.mat', mdict={'accuracy': accuracy})
