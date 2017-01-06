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

for i in range(200):
    model1 = FullyConnectedNet([100, 100], weight_scale=0.003, use_batchnorm = False, reg=0.6)
    solver1 = Solver(model1, data1,
                            print_every=data1['X_train'].shape[0], num_epochs=100, batch_size=100,
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
                            print_every=data1['X_train'].shape[0], num_epochs=100, batch_size=100,
                            update_rule='sgd',
                            optim_config={
                              'learning_rate': 0.03,
                            },verbose=False, lr_decay = 0.9,
                    )
    solver2.train()
    pass

    name = 'models/' + str(i) + '.pkl'
    with open(name, 'wb') as output:
        pickle.dump(model1, output, pickle.HIGHEST_PROTOCOL)
        pickle.dump(model2, output, pickle.HIGHEST_PROTOCOL)
