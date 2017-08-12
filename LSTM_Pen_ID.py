## NN models the PID controller
## Given input is the error and output is the control force
from __future__ import division
import numpy as np
from scipy.integrate import odeint
import random as random
import math as math
from copy import *
import pickle 
import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'   # removes warning
import tensorflow as tf
import matplotlib.pyplot as plt


## This define the dimension of input training sets
maxtime = 2
timestep = 0.1
num_steps = int(math.ceil(maxtime/timestep)) ## total length of the time series

## LSTM features
default_dict = {'state_size': 16, 'num_layer':2, 'batch_size': 50, 'epochs': 1000}
LSTM_dicts = {}
LSTM_dicts['default'] = default_dict.copy()

## Constructing multilayer NN
input_size = 1 
output_size = 1

## training data placeholder for LSTM
## Backpropoagaion length is num_steps
x_in = tf.placeholder(tf.float32, [None, num_steps, input_size])
x_unstack = tf.unstack(x_in, axis = 1) ## for static only
y_out = tf.placeholder(tf.float32, [None, num_steps, output_size])
y_unstack = tf.unstack (y_out, axis = 1) ## for static only
for key in LSTM_dicts:
    init_state = tf.placeholder(tf.float32, [LSTM_dicts[key]['num_layer'], 2, None, LSTM_dicts[key]['state_size']])


def make_weight (shape):
    return tf.Variable(tf.random_normal(shape), dtype = tf.float32, name = 'Wout')

def make_bias (shape):
    return tf.Variable(tf.zeros(shape), dtype = tf.float32, name = 'bout')

def LSTM_model (
    x = None, 
    y = None, 
    num_layer = None, 
    batch_size = None,
    state_size = None,
    input_size = input_size, 
    output_size = output_size,
    learning_rate = 0.005,
    **kwargs): 

    ## inialize the values for state
    batch_state = np.zeros((num_layer, 2, batch_size, state_size))
    ## unpack init_state for each layer in LSTM
    state_per_layer = tf.unstack(init_state, axis = 0)
    tuple_state = tuple([tf.contrib.rnn.LSTMStateTuple(state_per_layer[i][0], state_per_layer[i][1]) for i in range(num_layer)])
    Wout = make_weight([state_size, output_size])
    bout = make_bias([output_size]) 
    cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.BasicLSTMCell(state_size) for _ in range(num_layer)], state_is_tuple = True)
    ## Adding dropout wrapper
    # cell = tf.contrib.rnn.MultiRNNCell ([tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.BasicLSTMCell((state_size), state_is_tuple= True), 
                                            # input_keep_prob = 0.8) for _ in range(layer_num)], state_is_tuple= True)
    # cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob = 1)

    ## Dynamic API - recommended for computational efficiency
    ## state_series is the list of hidden states as tensors
    ## current_state is the final LSTM state tuple with hidden and cell state
    state_series , current_state = tf.nn.dynamic_rnn(cell, x, initial_state = tuple_state)
    output_series = tf.reshape(tf.matmul(tf.reshape(state_series, [-1, state_size]), Wout) + bout , [batch_size, num_steps, output_size] )

    ## output for plotting only
    output_plot = tf.reshape(tf.matmul(tf.reshape(state_series, [-1, state_size]), Wout) + bout , [1, num_steps, output_size] )
    cost = tf.reduce_mean((output_series - y)*(output_series- y))

    ## Static API -- not recommened, but more readable
    # state_series , current_state = tf.contrib.rnn.static_rnn(cell, x_unstack, initial_state = init_state)
    # output_series = [tf.matmul(state,Wout)+bout for state in state_series]
    # cost = tf.reduce_mean([(y_p - y_d)*(y_p - y_d) for y_p,y_d in zip(output_series, y_unstack) ])
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
    ## initial state for plotting
    plot_state = np.zeros((num_layer, 2, 1, state_size))  

    return { 'plot_state': plot_state,
             'batch_state': batch_state,
             'state_series': state_series,
             'output_series': output_series,
             'output_plot': output_plot,
             'Wout': Wout,
             'bout': bout,
             'cost': cost,
             'optimizer': optimizer}

for key in LSTM_dicts:
    print ('Making LSTM:' + str(key))
    LSTM_dicts[key].update(LSTM_model(x = x_in, y = y_out, **LSTM_dicts[key]))

## color map
cmap = plt.get_cmap('hsv')
## Training sample for LSTM
xtrainset = pickle.load(open('xtrainset.p','rb'))
ytrainset = pickle.load(open('ytrainset.p','rb'))
## Training the LSTM
sess=tf.Session()
sess.run(tf.global_variables_initializer())

for key in LSTM_dicts:
    epochs = LSTM_dicts[key]['epochs']
    batch_size = LSTM_dicts[key]['batch_size']
    cost_hist = [ ]
    for ind in range(epochs):
        batch_xs, batch_ys = xtrainset[batch_size*ind: batch_size*(ind+1)], ytrainset[batch_size*ind: batch_size*(ind+1)]
        batch_xs, batch_ys = np.reshape(batch_xs, [batch_size, num_steps, input_size]), np.reshape(batch_ys, [batch_size, num_steps, output_size])
        sess.run(LSTM_dicts[key]['optimizer'], feed_dict = {x_in: batch_xs, y_out: batch_ys, init_state: LSTM_dicts[key]['batch_state']})
        curr_cost = sess.run(LSTM_dicts[key]['cost'], feed_dict ={x_in: batch_xs, y_out: batch_ys, init_state: LSTM_dicts[key]['batch_state']})
        print (key + ' train cost:' + str(curr_cost))
        cost_hist.append(curr_cost)

    ## Save code - create saver and then add W and b, the trained variables to a collection
    my_vars = [x_in, init_state, LSTM_dicts[key]['Wout'], LSTM_dicts[key]['bout'], LSTM_dicts[key]['state_series'], LSTM_dicts[key]['output_plot'] , y_out]
    LSTM_saver = tf.train.Saver()
    for v in my_vars:
        tf.add_to_collection(str(key) + 'LSTMvars',v)
        LSTM_saver.save(sess, str(key) + 'LSTMmodel_save' )

    LSTM_dicts[key].update({'output_series_plot': np.squeeze(sess.run(LSTM_dicts[key]['output_plot'], 
                                feed_dict = {x_in: np.reshape(batch_xs[0],[1,num_steps,input_size]), init_state: LSTM_dicts[key]['plot_state']})),
        'LSTMtrain_cost': curr_cost,
        'LSTMcost_hist':cost_hist,
        'LSTM_color': cmap(np.random.uniform(0,1))
        })
sess.close()

print ("LSTM training results")
for key in LSTM_dicts:
    print ('LSTM:' + str(key))
    print ('LSTM train cost:' + str(LSTM_dicts[key]['LSTMtrain_cost']))

plt.figure(1)
for key in LSTM_dicts:
    plt.plot(LSTM_dicts[key]['LSTMcost_hist'], '-', color = LSTM_dicts[key]['LSTM_color'], label= key)


plt.xlabel('epoch')
plt.ylabel('LSTM train cost')
plt.title('LSTM train cost')
plt.legend()

plt.figure(2)
T = np.linspace(0, maxtime, num_steps)
plt.plot(T, batch_ys[0], 'k-', label = 'ture value')
for key in LSTM_dicts: 
    plt.plot(T, LSTM_dicts[key]['output_series_plot'],'-', color= LSTM_dicts[key]['LSTM_color'], label= key)
plt.xlabel('Time')
plt.ylabel('Angular position')
plt.title('LSTM time series prediction')
plt.legend()
plt.show()

