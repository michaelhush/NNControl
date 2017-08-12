 ## NN predicts the time series state (angular position) 
 ## Given a time-dependent external force and the initial position and velocity
from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint
import random as random
import math as math
from copy import *

import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'   # removes warning

import pickle

## time step and the length of the time series
timestep = 0.1
maxtime = 2
num_steps = int(math.ceil(maxtime/timestep))

## Create multi layer nerual net
input_size = num_steps
output_size= num_steps

## Training data input placeholder for NN
xNN = tf.placeholder(tf.float32,[None,input_size])
yNN_= tf.placeholder(tf.float32,[None,output_size])

## NN features
default_dict = {'hidden_dim':128, 'layers':['sigmoid'], 'epochs':1000, 'batch_size':50}
NN_dicts = {}

NN_dicts['default'] = default_dict.copy()
## Implement multi-layer NN
def make_weight (shape):
    return tf.Variable (tf.random_normal(shape))

def make_bias(shape):
    return tf.Variable(tf.zeros(shape))

def NN_model(
    x = None, 
    y = None, 
    layers = None, 
    input_size = num_steps, 
    output_size = num_steps, 
    hidden_dim = 1,
    learning_rate = 0.1,
    **kwargs):
    weight = [ ]
    bias = [ ]

    for index , item in enumerate(layers):
        bias.append(make_bias([hidden_dim]))
        if index == 0:
            weight.append(make_weight([input_size, hidden_dim]))
            curr_net = tf.add(tf.matmul(x, weight[index]), bias[index])
        else:
            weight.append(make_weight([hidden_dim, hidden_dim]))
            curr_net = tf.add(tf.matmul(curr_net, weight[index]), bias[index])
        if item == 'sigmoid':
            curr_net = tf.nn.sigmoid(curr_net) 
        elif item == 'relu':
            curr_net = tf.nn.relu(curr_net)
        elif item == 'tanh':
            curr_net = tf.nn.tanh(curr_net)
        else: 
            print('activation function invalid')
            raise ValueError
    weight.append(make_weight([hidden_dim, output_size]))
    bias.append(make_bias([output_size]))
    y_pred = tf.add(tf.matmul(curr_net, weight[-1]), bias[-1])

    cost = tf.reduce_mean((y_pred - y)*(y_pred - y))
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

    return {'weight':weight,
            'bias':bias,
            'y_pred':y_pred,
            'cost':cost,
            'optimizer':optimizer}


for key in NN_dicts:

    print ('Making NN:' + str(key))
    NN_dicts[key].update(NN_model(x = xNN, y = yNN_, input_size = num_steps, output_size = num_steps, **NN_dicts[key]))



## color map
cmap = plt.get_cmap('hsv')
## training sample for KNN
xtrainset = pickle.load(open('xtrainset.p','rb'))
ytrainset = pickle.load(open('ytrainset.p','rb'))

sess = tf.Session()
sess.run (tf.global_variables_initializer())


for key in NN_dicts:
    epochs = NN_dicts[key]['epochs']
    batch_size = NN_dicts[key]['batch_size']
    cost_hist = [ ]
    for ind in range(epochs):
        batch_xs, batch_ys = xtrainset[batch_size*ind: batch_size*(ind+1)], ytrainset[batch_size*ind: batch_size*(ind+1)]
        sess.run(NN_dicts[key]['optimizer'], feed_dict= {xNN: batch_xs, yNN_: batch_ys})
        curr_cost = sess.run(NN_dicts[key]['cost'], feed_dict= {xNN: batch_xs, yNN_: batch_ys})
        print(key + ' train cost :' + str(curr_cost) )
        cost_hist.append(curr_cost)

    ## Saving code - create saver and then add W and b, the trained variables to a collection
    my_vars = [xNN, NN_dicts[key]['y_pred'], yNN_]
    saver = tf.train.Saver()
    for v in my_vars:
        tf.add_to_collection(str(key)+'NNvars', v)
        saver.save(sess, str(key) + 'NNmodel_save.ckpt')
    
    NN_dicts[key].update({'y_pred_plot': np.squeeze(sess.run(NN_dicts[key]['y_pred'], feed_dict= {xNN: np.reshape(batch_xs[0],[1,num_steps])})),
        'NNtrain_cost': curr_cost,
        'NNcost_hist': cost_hist,
        'NN_color': cmap(np.random.uniform(0,1))})
    ## pickle NN_dict
    # pickle.dump(NN_dicts, open(str(key)+"NN_dicts.p","wb"))

sess.close()

## TODO: pickle NN_dict and defer this code
print("NN Training results")

for key in NN_dicts:
    print('NN:' + str(key))
    print('NN train cost:' + str(NN_dicts[key]['NNtrain_cost']))

plt.figure(1)
for key in NN_dicts:
    plt.plot(NN_dicts[key]['NNcost_hist'], '-', color= NN_dicts[key]['NN_color'], label = key)

plt.xlabel('epoch')
plt.ylabel('NN train cost')
plt.title('NN train cost')
plt.legend()

plt.figure(2)
T = np.linspace(0, maxtime, num_steps)
plt.plot(T, batch_ys[0], 'k-', label = 'ture value')
for key in NN_dicts: 
    plt.plot(T, NN_dicts[key]['y_pred_plot'],'-', color= NN_dicts[key]['NN_color'], label= key)
plt.xlabel('Time')
plt.ylabel('Angular position')
plt.title('NN time series prediction')
plt.show()
