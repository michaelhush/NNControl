## Control algorith -- greedy
## An NN controller for a speicific plant with input noise (disturbance)
from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as si
import random as random
import math
from copy import *
import pickle


import tensorflow as tf
import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'   # removes warning



class BaseCell(tf.contrib.rnn.RNNCell):
    
    def __init__(self,state_size,output_size, input_size, state_is_tuple = True, dtype=tf.float64, act_func = tf.tanh):
        self.dtype = dtype
        self._output_size = output_size
        self.input_size = input_size
        self.cell = tf.contrib.rnn.BasicRNNCell(state_size, activation=act_func)
        self._state_size = self.cell.state_size
        self.W_out = tf.Variable(tf.zeros([self._state_size, self.output_size], dtype = self.dtype) , name = 'Wout')
        self.b_out = tf.Variable(tf.zeros([self.output_size], dtype = self.dtype), dtype = self.dtype, name = 'bout')
    
    def __call__(self, inputs, state, scope=None):
        """Run this RNN cell on inputs, starting from the given state.
        Args:
          inputs: `2-D` tensor with shape `[batch_size x input_size]`.
          state: if `self.state_size` is an integer, this should be a `2-D Tensor`
            with shape `[batch_size x self.state_size]`.  Otherwise, if
            `self.state_size` is a tuple of integers, this should be a tuple
            with shapes `[batch_size x s] for s in self.state_size`.
          scope: VariableScope for the created subgraph; defaults to class name.
        Returns:
          A pair containing:
          - Output: A `2-D` tensor with shape `[batch_size x self.output_size]`.
          - New state: Either a single `2-D` tensor, or a tuple of tensors matching
                the arity and shapes of `state`.
        """
        (out_state,n_state) = self.cell.__call__(inputs, state, scope=scope)
        n_output = tf.matmul(out_state, self.W_out) + self.b_out
        return (n_output,n_state)
    
    @property
    def state_size(self):
        return self._state_size
    
    @property
    def output_size(self):
        return self._output_size

#The class for the feedback nerual net
#most of the functions and setting for the differential equation solver are packed in this class because it is convenient 
class FBCell(tf.contrib.rnn.RNNCell):

    def damped_pen (self,state,t): 
        (x, p, u) = tf.unstack(state)
        dx = p
        dp =  self.g * tf.sin(x) - self.mu * p  + u
        du = 0
        return tf.stack([dx,dp,du])

    def __init__(self,cell_cont, timestep):
        self.g = 9.8
        self.mu = 0.1
        self.ode_dt = tf.constant([0.0,timestep])
        self.cell_cont = cell_cont
        self.ss_cont = self.cell_cont.state_size
        self.ss_ode = 2
        self._state_size = (self.ss_cont, self.ss_ode, self.cell_cont.output_size)
        self._output_size = self.cell_cont.input_size
        
    def __call__(self, inputs, state, scope=None):
        (ls_cont,ls_ode, l_output) = state
        in_ode = inputs + l_output
        total_ode = tf.concat(axis=1,values=[ls_ode, in_ode])
        temp_ode = []
        for cs_ode in tf.unstack(total_ode):
            temp_ode.append(tf.contrib.integrate.odeint(self.damped_pen,cs_ode,self.ode_dt)[-1,0:2])
        temp_ode = tf.stack(temp_ode)
        ns_ode = temp_ode
        out_ode = temp_ode[:,0:1]
        (out_cont, ns_cont) = self.cell_cont.__call__(out_ode,ls_cont)
        return (out_ode, (ns_cont, ns_ode, out_cont))   
        
    @property
    def state_size(self):
        return self._state_size
    
    @property
    def output_size(self):
        return self._output_size

#total time for each episode
epitime = 2.0
#time step between each sample
timestep = 0.1
#number of total time steps
num_steps = int(math.ceil(epitime/timestep))
#data type to be used by all NN
dtype = tf.float64
#size of each batch    
batch_size = 16
#state size of RNN
state_size = 32
#activation function
act_func = tf.tanh
#max learning rate for ADAMS optimizer
learning_rate = 0.01
#number of episodes to train for 
episodes = 100
#controller inputs
input_size = 1 
#controller outputs
output_size = 1 
#The control target
setout = tf.constant([0.0], dtype = dtype)
#noise standard deviation
noise_sigma = 1.0


#define function for generating noise based on parameters
def gen_noise():
    temp = [ np.full([num_steps,output_size], np.random.normal(0, noise_sigma)) for _ in range(batch_size)]
    return np.array(temp)
    # return self.nF*np.random.randn([self.num_steps, self.plant_input_size])


#cost history storage
cost_hist = [ ]

sess = tf.Session()

print ('Constructing variables...')

#Construct the RNN cell for the controller
cell_cont = BaseCell(state_size, output_size, input_size, dtype = dtype, act_func = act_func)

#Construct the RNN cell for the feedback
cell_fb = FBCell(cell_cont,timestep)

#extract some relevant properties from the fb_cell
#size of variables for the ode
ode_size = cell_fb.ss_ode 

#Input to feedback (which is the size of the controller output)
fb_in = tf.placeholder(dtype, [batch_size, num_steps, output_size])
#Initial state of ode
fb_diff = tf.placeholder(dtype, [batch_size, output_size])
#Initial state of controller
state_cont = tf.placeholder(dtype, [batch_size, state_size])
#Initial state of ode
state_ode = tf.placeholder(dtype, [batch_size, ode_size])
#Initiall state of feedback cell
fb_init_state = (state_cont,state_ode,fb_diff)

#Dynamically unroll the RNN cellf or the feedback
fb_pred, tf_state_tuple = tf.nn.dynamic_rnn(cell_fb, fb_in, initial_state = fb_init_state)

#extract some relevant properties from the fb_cell

# zero arrays 
zeros_batch_cont = np.zeros([batch_size, state_size])
zeros_batch_ode = np.zeros([batch_size, ode_size])
zeros_batch_output = np.zeros([batch_size, output_size])

print ('Setting up optimizer...')

#Set the cost of performance for the feedback, basically the output should be at the set value
fb_cost = tf.reduce_mean((fb_pred - setout)*(fb_pred - setout))
#Set up the optimization tests
fb_optimizer = tf.train.AdamOptimizer(learning_rate).minimize(fb_cost)

print ('Initializing variables...')

sess.run(tf.global_variables_initializer())

print ('Training controller...')

cost_save_skip = 10

for epo in range(episodes):
    print ('Episodes:{}'.format(epo + 1))
    
    noise_batch = gen_noise()
    
    sess.run(fb_optimizer, feed_dict = {
        fb_in: noise_batch,
        state_cont:zeros_batch_cont, 
        state_ode:zeros_batch_ode,
        fb_diff:zeros_batch_output})
    
    if epo%cost_save_skip==0:
        curr_cost = sess.run(fb_cost, feed_dict = {
            fb_in: noise_batch,
            state_cont:zeros_batch_cont, 
            state_ode:zeros_batch_ode,
            fb_diff:zeros_batch_output})

        cost_hist.append(curr_cost)
    
print ('Final train cont cost:{}'.format(curr_cost))

print('Making plots...')

plt.plot(range(len(cost_hist)), cost_hist, '-', label = 'controller costs')
plt.title('Controller cost history')
plt.legend()   
plt.show()

print('Done.')