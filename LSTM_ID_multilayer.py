## NN models the PID controller
## Given input is the error and output is the control force
from __future__ import division
import numpy as np
from scipy.integrate import odeint
import random as random
import math as math
from copy import *

import tensorflow as tf
import matplotlib.pyplot as plt


## This define the dimension of input training sets
maxtime = 2
timestep = 0.1
steps_num = int(maxtime/timestep) ## total length of the time series

## damp pendulum parameters
g = 9.8
mu = 0.1

## Pendulum properties
## Angular position range

## External force mean and standard deviation
mu, sigma = 0, 5*math.sqrt(steps_num)

## initial position of pendulum
# int_y1 = 0.0 
# int_y2 = 0.0

## Constructing multilayer NN
input_num = 1 
output_num = 1
state_num = 16
layer_num = 2
batch_num = 32

############################################################################
## Generate training set
def nonlinear_damped_pen_position (y,t,g,mu,u):
    y1,y2 = y
    dydt = [y2, -g*np.sin(y1) - mu*y2 + u]
    return dydt

## Generate training set
def linear_damped_pen_position (y,t,g,mu,u):
    y1,y2 = y
    dydt = [y2, -g*y1 -  mu*y2 + u]
    return dydt

## Nonlinear model 
def nonlinear_update (Xsample):
    t=[0,timestep]
    yv_sol = odeint (nonlinear_damped_pen_position,
                     [Xsample[0], Xsample[1]], t, args=(g,mu, Xsample[2]))
    y_next, v_next = yv_sol[-1]  
    Ysample = [y_next, v_next]
    return Ysample

## Linear approximation (benchmarg against)
def linear_update (Xsample):
    t=[0,timestep]
    yv_sol = odeint (linear_damped_pen_position,
                    [Xsample[0], Xsample[1]], t, args=(g,mu, Xsample[2]))
    y_next, v_next = yv_sol[-1]  
    Ysample = [y_next, v_next]
    return Ysample

## Generating training and testing samples
def training_batch (number_sample):    
    batch_xs = []
    batch_ys = []  
    int_y1 = random.uniform(0, math.pi)
    int_y2 = random.uniform(0,0)
    batch_state = np.full([layer_num, 2, batch_num, state_num], int_y1)
    # batch_state = np.zeros((layer_num, 2, batch_num, state_num))
    for _ in range(number_sample):   
        ## Xsample for time-varying input force
        samp_xs = [np.random.normal(mu,sigma) for i in xrange(steps_num) ]
        currY = [int_y1,int_y2]
        samp_ys  = [ ]
        # Sovling the ODE
        for force in samp_xs:
            nextY = nonlinear_update ( currY + [force] )
            currY = nextY
            samp_ys.append(currY[0])
        batch_xs.append(copy(samp_xs))
        batch_ys.append(copy(samp_ys))  
    batch_xs = np.array(batch_xs)
    batch_ys = np.array(batch_ys)
    return batch_xs, batch_ys, batch_state

def testing_batch (number_sample): 
    batch_xs, batch_ys, batch_state = training_batch(number_sample)
    batch_yl = []
    int_y1 = random.uniform(0, math.pi)
    int_y2 = random.uniform(0,0)
    batch_state = np.full([layer_num, 2, batch_num, state_num], int_y1)
    for samp_xs in batch_xs:
        # Solving the ODE 
        currY = [int_y1,int_y2]
        samp_yl  = [ ]
        for force in samp_xs:
            nextY = linear_update ( currY + [force] )
            currY = nextY
            samp_yl.append(currY[0])
        batch_yl.append(copy(samp_yl))
    batch_yl = np.array(batch_yl)
    return batch_xs, batch_ys, batch_yl, batch_state

#######################################################################################


init_state = tf.placeholder(tf.float32, [layer_num, 2, None, state_num])
## Unpack the init_state for each layer in LSTM
state_per_layer = tf.unstack(init_state, axis=0)
tuple_state = tuple ([tf.contrib.rnn.LSTMStateTuple(state_per_layer[i][0], state_per_layer[i][1]) for i in range(layer_num)])

## Backpropoagaion length is steps_num

## use x_in for dynamic API 
x_in = tf.placeholder(tf.float32, [None, steps_num, input_num])
## unstack x_in into steps_num tensors each with shape [None, input_num] for static API 
x_unstack = tf.unstack (x_in, axis= 1)

## Use y_out for dynamic API
y_out = tf.placeholder(tf.float32, [None,steps_num,output_num])
## Use unstack for static 
y_unstack = tf.unstack(y_out, axis=1)

Wout= tf.Variable(tf.random_normal([state_num,output_num]),dtype= tf.float32)  
bout = tf.Variable(tf.zeros([output_num]),dtype= tf.float32)

## Forward passes
# def 


# cell = tf.contrib.rnn.MultiRNNCell ([tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.BasicLSTMCell((state_num), state_is_tuple= True), input_keep_prob = 1) for _ in range(layer_num)], state_is_tuple= True)
# cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob = 1)
# cell = tf.contrib.rnn.BasicLSTMCell(state_num, state_is_tuple = True)
cell = tf.contrib.rnn.MultiRNNCell( [tf.contrib.rnn.BasicLSTMCell(state_num) for _ in range(layer_num)] ,state_is_tuple= True)



## Static API
# state_series , current_state = tf.contrib.rnn.static_rnn(cell, x_unstack, initial_state = init_state)
# output_series = [tf.matmul(state,Wout)+bout for state in state_series]
# cost = tf.reduce_mean([(y_p - y_d)*(y_p - y_d) for y_p,y_d in zip(output_series, y_unstack) ])

## Dynamic API 
## state_series is the list of hidden states as tensors
## current_state is the final LSTM state tuple with hidden and cell state
state_series , current_state = tf.nn.dynamic_rnn(cell, x_in, initial_state = tuple_state)

output_series = tf.reshape(tf.matmul(tf.reshape(state_series, [-1, state_num]), Wout) +bout , [batch_num, steps_num, output_num] )
cost = tf.reduce_mean((output_series - y_out)*(output_series- y_out))

tf.summary.FileWriter('Multilayer_LSTM', graph = tf.get_default_graph())  

################ Traning 
# global_step = tf.Variable(0,trainable=False)
# starter_learning_rate=0.5

# learning_rate=tf.train.exponential_decay(starter_learning_rate, global_step,10000,
#                                         0.98,staircase=True)
train_step = tf.train.AdamOptimizer(0.005).minimize(cost)

# train_step = tf.train.AdagradOptimizer(0.3).minimize(cost)

runs = 1000
sess=tf.Session()
sess.run(tf.global_variables_initializer())
cost_hist = [ ]

## Compare nonlinear vs linear vs NN
yl_ = tf.placeholder(tf.float32,[None, steps_num, output_num])
y_ = tf.placeholder(tf.float32,[None, steps_num, output_num])
lin_cost = tf.reduce_mean((yl_ - y_)*(yl_ - y_))

print("Training NN")
for index in range(runs):
    print(index)
    batch_xs, batch_ys, batch_state = training_batch (batch_num)
    batch_xs = np.reshape(batch_xs, [batch_num, steps_num, input_num])
    batch_ys = np.reshape(batch_ys, [batch_num, steps_num, output_num])
    sess.run(train_step ,feed_dict={x_in: batch_xs, y_out:batch_ys, init_state : batch_state})
    cost_hist.append(sess.run(cost, feed_dict={x_in: batch_xs, y_out: batch_ys, init_state: batch_state}))
    print (sess.run(cost, feed_dict = {x_in: batch_xs, y_out: batch_ys, init_state: batch_state}))
cost_hist = np.array(cost_hist)
cost_num = np.arange(cost_hist.size)

#################################################################################################


## testing 
test_num = 100

batch_xt, batch_yt, batch_ylt, batch_state = testing_batch(test_num)
batch_xt = np.reshape(batch_xt, [test_num, steps_num, input_num])
batch_yt = np.reshape(batch_yt, [test_num, steps_num, output_num])
batch_ylt = np.reshape (batch_ylt,[test_num, steps_num, output_num])

batch_state = np.zeros((layer_num, 2, test_num, state_num))

output_series = tf.reshape(tf.matmul(tf.reshape(state_series, [-1, state_num]), Wout) +bout , [test_num, steps_num, output_num] )
cost = tf.reduce_mean((output_series - y_out)*(output_series- y_out))

test_linear_cost = sess.run(lin_cost, feed_dict={yl_: batch_ylt, y_: batch_yt})
test_NN_cost = sess.run(cost, feed_dict={x_in: batch_xt, y_out: batch_yt, init_state: batch_state})
print("NN vs linear results")
print("NN final cost:")
print(test_NN_cost)
print("linear model cost:")
print(test_linear_cost)

## Plot results
plt.figure(1)
plt.plot(cost_num,cost_hist,'bo')  
plt.title("Plot of NN and linear cost function ")
plt.show()
