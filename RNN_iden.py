## NN models the PID controller
## Given input is the error and output is the control force
from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint
import random as random
import math as math
from copy import *

import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'   # removes warning

import tensorflow as tf
import matplotlib.pyplot as plt
import pickle
import batch_gen


## This define the dimension of input training sets
maxtime = 2
timestep = 0.1
num_steps = int(math.ceil(maxtime/timestep)) ## total length of the time series (TF length for backprop)
backprop_length = 4 ## truncated backprop length

## RNN features
default_dict = {'state_size': 2, 'hidden_dim': 256, 'layers':[], 'batch_size':50, 'epochs':1000 }
RNN_dicts = {}
RNN_dicts['default'] = default_dict.copy()

## Constructing multilayer NN
input_size = 1 
output_size = 1
## input and true output placeholders
x_in = tf.placeholder(tf.float32, [None, num_steps, input_size])


y_out = tf.placeholder(tf.float32, [None,num_steps,output_size])


for key in RNN_dicts:
    init_state = tf.placeholder(tf.float32,[None, RNN_dicts[key]['state_size']])

def make_weight(shape):
    return tf.Variable(tf.random_normal(shape), dtype = tf.float32)

def make_bias(shape):
    return tf.Variable(tf.random_normal(shape), dtype = tf.float32)

def RNN_model (
    x = None,
    y = None, 
    layers = None, 
    hidden_dim = 1, 
    input_size = input_size,
    output_size = output_size,
    state_size = None, 
    batch_size = None, 
    learning_rate = 0.01,
    **kwargs
    ):
    concat_num = input_size + state_size
    x_unstack = tf.unstack(x, axis = 1)
    y_unstack = tf.unstack(y, axis = 1)
    weight = [ ]
    bias = [ ]
    state_series = [ ]
    output_series = [ ]
    current_state = init_state
    if len(layers) == 0:
        W_in = make_weight([concat_num, state_size])
        b_in = make_bias([state_size])
    else: 
        W_in = make_weight([concat_num, hidden_dim])
        b_in = make_bias([hidden_dim])

    W_state = make_weight([hidden_dim, state_size])
    b_state = make_bias([state_size])
    W_out = make_weight([state_size, output_size])
    b_out = make_bias([output_size])
    for index, item in enumerate(layers):
        if index == 0:
            pass
        else: 
            weight.append(make_weight([hidden_dim, hidden_dim]))
            bias.append(make_bias([hidden_dim]))
            if item == 'sigmoid':
                curr_net = tf.nn.sigmoid(curr_net)
            elif item == 'relu':
                curr_net = tf.nn.relu(curr_net)
            elif item == 'tanh':
                curr_net = tf.nn.tanh(curr_net)
            else:
                print ('activation function invalid')
                raise ValueError

    for current_x in x_unstack:
        in_and_state_concat = tf.concat([current_x, current_state], axis = 1)
        curr_net = tf.add(tf.matmul(in_and_state_concat, W_in), b_in)
        for index, item in enumerate(layers):
            if index == 0 :
                pass
            else: 
                curr_net = tf.matmul(curr_net, weight[index]) + bias[index]
                if item == 'sigmoid':
                    curr_net = tf.nn.sigmoid(curr_net)
                elif item == 'tanh': 
                    curr_net = tf.nn.tanh(curr_net)
                elif item == 'relu':
                    curr_net = tf.nn.relu(curr_net)
                else:
                    print ('activation function invalid')
                    raise ValueError
        ## state layer
        if len(layers) == 0:
            diff_element = tf.nn.tanh(curr_net)
        else: 
            diff_element = tf.nn.tanh(tf.add(tf.matmul(curr_net, W_state), b_state))
        next_state = current_state + diff_element
        state_series.append(next_state)
        ## output layer
        current_output = tf.matmul(next_state, W_out) + b_out
        output_series.append(current_output)
        current_state = next_state
    
    ## define cost function
    cost = tf.reduce_mean([ (y_p - y_d)*(y_p - y_d) for y_p, y_d in zip (output_series, y_unstack)] ) 
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
    ## initialize init_state
    batch_state = np.zeros([batch_size, state_size], dtype = np.float32)
    return { 'state_series': state_series,
             'output_series': output_series,
             'cost': cost,
             'optimizer': optimizer,
             'batch_state': batch_state}

for key in RNN_dicts:
    print ('Making RNN:' + str(key))
    RNN_dicts[key].update(RNN_model(x = x_in, y = y_out, input_size = input_size, output_size = output_size, **RNN_dicts[key]))

## color map
cmap = plt.get_cmap('hsv')
## Training sample for RNN
xtrainset = pickle.load(open('xtrainset.p','rb'))
ytrainset = pickle.load(open('ytrainset.p','rb'))
## Training the RNN
sess=tf.Session()
sess.run(tf.global_variables_initializer())

for key in RNN_dicts:
    epochs = RNN_dicts[key]['epochs']
    batch_size = RNN_dicts[key]['batch_size']
    cost_hist = [ ] 
    for ind in range(epochs):
        batch_xs, batch_ys = xtrainset[batch_size*ind: batch_size*(ind+1)], ytrainset[batch_size*ind: batch_size*(ind+1)]
        batch_xs, batch_ys = np.reshape(batch_xs, [batch_size, num_steps, input_size]), np.reshape(batch_ys, [batch_size, num_steps, output_size])
        sess.run(RNN_dicts[key]['optimizer'], feed_dict = {x_in: batch_xs, y_out: batch_ys, init_state: RNN_dicts[key]['batch_state']})
        curr_cost = sess.run(RNN_dicts[key]['cost'], feed_dict ={x_in: batch_xs, y_out: batch_ys, init_state: RNN_dicts[key]['batch_state']})
        print (key + ' train cost:' + str(curr_cost))
        cost_hist.append(curr_cost)

    my_vars = [x_in, init_state, tf.convert_to_tensor(RNN_dicts[key]['output_series']), RNN_dicts[key]['cost'],  y_out] ## convert list to tensor for storing
    RNN_saver = tf.train.Saver()
    for v in my_vars:
        tf.add_to_collection(str(key) + 'RNNvars', v)
        RNN_saver.save(sess, str(key) + 'RNNmodel_save')

    RNN_dicts[key].update({
        'RNNtrain_cost': curr_cost,
        'RNNcost_hist': cost_hist,
        'RNN_color': cmap(np.random.uniform(0,1))
        })


print ("RNN training results")
for key in RNN_dicts:
    print ('RNN:' + str(key))
    print ('RNN train cost:' + str(RNN_dicts[key]['RNNtrain_cost']))

plt.figure(1)
for key in RNN_dicts:
    plt.plot(RNN_dicts[key]['RNNcost_hist'], '-', color = RNN_dicts[key]['RNN_color'], label= key)


plt.xlabel('epoch')
plt.ylabel('RNN train cost')
plt.title('RNN train cost')

plt.figure(2)
T = np.linspace(0, maxtime, num_steps)
plt.plot(T, batch_ys[0], 'k-', label = 'true value')
for key in RNN_dicts:
    output_series_plot = sess.run(RNN_dicts[key]['output_series'], feed_dict = {x_in: batch_xs, y_out: batch_ys, init_state: RNN_dicts[key]['batch_state']})
    output_series_plot = np.array(output_series_plot)
    plt.plot(T , output_series_plot[:,0], '-', color = RNN_dicts[key]['RNN_color'], label = key)

plt.xlabel('Time')
plt.ylabel('Angular position')
plt.title('RNN time series prediction')
plt.legend()

sess.close()

plt.show()


