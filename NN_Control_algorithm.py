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



class SOCell(tf.contrib.rnn.LSTMCell):
    
    def __init__(self,state_size,output_size, state_is_tuple = True, dtype=tf.float64):
        self.dtype = dtype
        self._output_size = output_size
        self.cell = tf.contrib.rnn.LSTMCell(state_size)
        self._state_size = self.cell.state_size
        self._hs_size = state_size 
        self.W_out = tf.Variable(tf.zeros([self._hs_size, self.output_size], dtype = self.dtype) , name = 'Wout')
        self.b_out = tf.Variable(tf.zeros([self.output_size], dtype = self.dtype), dtype = self.dtype, name = 'bout')
    
    def __call__(self, inputs, state, scope=None):
        (out_state,n_state) = self.cell.__call__(inputs, state, scope=scope)
        n_output = tf.matmul(out_state, self.W_out) + self.b_out
        #self.plant_pred = tf.einsum('ijk,kl->ijl', out_state, W_out) + b_out
        return (n_output,n_state)
    
    @property
    def state_size(self):
        return self._state_size
    
    @property
    def output_size(self):
        return self._output_size


class FBCell(tf.contrib.rnn.LSTMCell):
    
    def __init__(self,cell_plant,cell_cont):
        self.cell_plant = cell_plant
        self.cell_cont = cell_cont
        self.ss_plant = self.cell_plant.state_size
        self.ss_cont = self.cell_cont.state_size
        # self._state_size = (self.ss_plant[0],self.ss_plant[1],self.ss_cont[0],self.ss_cont[1],1)
        self._state_size = (self.ss_plant[0],self.ss_plant[1],self.ss_cont[0],self.ss_cont[1],self.cell_cont.output_size)
        self._output_size = self.cell_plant.output_size
        
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
        (lcs_plant,lhs_plant,lcs_cont,lhs_cont,l_output) = state
        in_plant = inputs + l_output
        (out_plant, (ncs_plant,nhs_plant)) = self.cell_plant.__call__(in_plant,(lcs_plant,lhs_plant))
        (out_cont, (ncs_cont,nhs_cont)) = self.cell_cont.__call__(out_plant,(lcs_cont,lhs_cont))
        return (out_plant,(ncs_plant, nhs_plant, ncs_cont, nhs_cont, out_cont))
        
    @property
    def state_size(self):
        return self._state_size
    
    @property
    def output_size(self):
        return self._output_size

class Model(object):
    def select_model(self, name):
        if name == 'Pendulum': return Pendulum()
        # if name == 'Quantum' : return Quantum() 
        assert 0, 'Model name does not exist' + name 

class Pendulum(Model):
    def __init__(self):
        self.config_dynamics()

    def config_dynamics(self, g=9.8, mu=0.1):
        self.g = g #pendumlem weight
        self.mu = mu #pendulem dampening


    def dynamics(self,state,t,u): 
        x, p = state
        dx = p
        dp =  -self.g * np.sin(x) - self.mu * p  + u
        return [dx,dp]

    def variables_name(self):
        return ["pos x","pos y","ang mom"]

    def real_state_to_response(self, real_state):
    ## Transforming angular position to x,y coordinate
        return np.array([np.sin(real_state[0]),-np.cos(real_state[0]),real_state[1]])

## TODO: add quantum simulation here, note it must have the same function name 'config_dynamics' and 'dynamics'
# class Quantum(Model):
#     def __init__(self):
#         self.config_dynamics()
#     def config_dynamics(self, ):
#         pass
#     def dynamics(self, ):
#         pass

class LSTM_controller(object):
    def __init__(self, epitime = 1, timestep = 0.1, model_name = 'Pendulum'):
        self.epitime = epitime
        self.timestep = timestep
        self.t_arr = np.array([0.0,self.timestep]) # For using odeint
        self.num_steps = int(math.ceil(self.epitime/self.timestep))
        ## Allow users to define multiple dynamical models and train controller for that model
        self.model_name = model_name
        self.model = Model().select_model(model_name)
        self.outvar_names = self.model.variables_name()
        self.dtype = tf.float64
        self.sqrttimestep = np.sqrt(self.timestep)
        self.batch_count = 0
        self.data_count = 0


        self.sess = tf.Session()
        
        #run default configurations
        self.config_noise()
        
        #data storage
        self.force_data = [ ] 
        self.pos_data = [ ] 
        self.noise_data = [ ]
        
        #cost history storage
        self.plant_hist = [ ]
        self.cont_hist = [ ]
        
    def config_noise(self, nF = 5):
        self.nF = nF #magnitude of noise 
    
    ## TODO: potentially create multiple layers LSTM
    def config_lstm(self, batch_size = 128, plant_state_size = 128, cont_state_size = 32, input_size = 1, output_size = 3, learning_rate = 0.001, setout=None):
        self.batch_size = batch_size
        self.plant_state_size = plant_state_size
        self.cont_state_size = cont_state_size
        self.plant_input_size = input_size
        self.plant_output_size = output_size
        self.cont_input_size = self.plant_output_size #controller input must be plant output
        self.cont_output_size = self.plant_input_size #controller ouptut must be plant input
        self.learning_rate = learning_rate
        
        ## If setout is None set target to bring all outputs to zero
        if setout is None:
            self.setout = tf.constant(np.zeros(self.plant_output_size), dtype = self.dtype)
        else:
            self.setout = tf.constant(setout, dtype = self.dtype)
        
        ## plant variables
        self.plant_in = tf.placeholder(self.dtype, [None, self.num_steps, self.plant_input_size])
        self.plant_true = tf.placeholder(self.dtype, [None, self.num_steps, self.plant_output_size])
        self.cstate_plant = tf.placeholder(self.dtype, [None, self.plant_state_size])
        self.hstate_plant = tf.placeholder(self.dtype, [None, self.plant_state_size])
        with tf.variable_scope('plant'):
            self.plant_init_state = tf.contrib.rnn.LSTMStateTuple(self.cstate_plant, self.hstate_plant)
            self.cell_plant = SOCell(self.plant_state_size,self.plant_output_size, dtype = self.dtype)
            self.plant_pred, self.plant_state_tuple = tf.nn.dynamic_rnn(self.cell_plant, self.plant_in, initial_state = self.plant_init_state)
        self.plant_var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,'plant')
        
        ## controller variables
        self.cont_in = tf.placeholder(self.dtype, [None, 1, self.cont_input_size])
        self.cstate_cont = tf.placeholder(self.dtype, [None, self.cont_state_size])
        self.hstate_cont = tf.placeholder(self.dtype, [None, self.cont_state_size])
        with tf.variable_scope('controller'):
            self.cont_init_state = tf.contrib.rnn.LSTMStateTuple(self.cstate_cont, self.hstate_cont)
            self.cell_cont = SOCell(self.cont_state_size,self.cont_output_size, dtype = self.dtype)
            self.cont_pred_tuple = tf.nn.dynamic_rnn(self.cell_cont, self.cont_in, initial_state = self.cont_init_state)
        self.cont_var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,'controller')
        
        self.fb_in = tf.placeholder(self.dtype, [None, self.num_steps, self.plant_input_size])
        self.fb_diff = tf.placeholder(self.dtype,[None, self.plant_input_size])
        self.fb_init_state = (self.cstate_plant,self.hstate_plant,self.cstate_cont,self.hstate_cont,self.fb_diff)
        with tf.variable_scope('feedback'):
            self.cell_fb = FBCell(self.cell_plant,self.cell_cont)
            self.fb_pred, self.tf_state_tuple = tf.nn.dynamic_rnn(self.cell_fb, self.fb_in, initial_state = self.fb_init_state)
        self.fb_var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'feedback')

        # step 2 - fitting plant to data: cost and optimizer
        self.plant_cost = tf.reduce_mean((self.plant_pred - self.plant_true)*(self.plant_pred - self.plant_true))
        self.plant_optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.plant_cost, var_list=self.plant_var_list)
        
        # step 3 - fitting controller to plant and target: cost and optimizer
        self.fb_cost = tf.reduce_mean((self.fb_pred - self.setout)*(self.fb_pred - self.setout))
        # self.fb_cost = tf.reduce_mean(self.fb_pred*self.fb_pred)
        self.fb_optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.fb_cost, var_list= [self.cont_var_list, self.fb_var_list])
        
        # zero arrays 
        self.zeros_one_plant_state = np.zeros([1, self.plant_state_size])
        self.zeros_one_cont_state = np.zeros([1, self.cont_state_size])
        self.zeros_batch_plant_state = np.zeros([self.batch_size, self.plant_state_size])
        self.zeros_batch_cont_state = np.zeros([self.batch_size, self.cont_state_size])
        self.zeros_batch_plant_in = np.zeros([self.batch_size, self.plant_input_size])
        self.zeros_one_plant_in = np.zeros([1, self.plant_input_size])
    
    def gen_noise(self):
        temp = np.random.normal(0, self.nF)
        return np.full([self.num_steps, self.plant_input_size], temp)
        # return self.nF*np.random.randn([self.num_steps, self.plant_input_size])
    
    def gen_init_state(self):
        return np.array([0.0, 0.0])
        #return np.array([np.random.uniform()* 2*np.pi, np.random.normal(0, 1.0)])
    
    def initialize_variables(self):
        self.sess.run(tf.global_variables_initializer())
    
    def add_fpn_data(self, n, f, p):
        self.noise_data.append(n)
        self.force_data.append(f)
        self.pos_data.append(p)
        self.data_count += 1
      
    def shuffle_n_data(self):
        noise = self.noise_data[:]
        random.shuffle(noise)
        self.noise_shuffle = [ ]
        self.noise_shuffle = np.array(noise)

    def shuffle_fp_data(self):
        self.force_shuffle = [ ]
        self.pos_shuffle = [ ]
        combined = list(zip(self.force_data, self.pos_data))
        random.shuffle(combined)
        self.force_shuffle[:], self.pos_shuffle[:] = zip(*combined)  
        self.force_shuffle = np.array(self.force_shuffle)
        self.pos_shuffle = np.array(self.pos_shuffle)

    def get_fp_batch(self):
        self.batch_count += self.batch_size
        if self.batch_count >= self.data_count:
            self.shuffle_fp_data()
            self.batch_count = self.batch_size
        
        force_batch = self.force_shuffle[self.batch_count- self.batch_size: self.batch_count]
        resp_batch = self.pos_shuffle[self.batch_count - self.batch_size: self.batch_count]

        force_batch = np.reshape(force_batch, [self.batch_size, self.num_steps, self.plant_input_size])
        resp_batch = np.reshape(resp_batch, [self.batch_size, self.num_steps, self.plant_output_size])
        return force_batch, resp_batch

    def get_n_batch(self):
        self.batch_count += self.batch_size
        if self.batch_count >= self.data_count:
            self.shuffle_n_data()
            self.batch_count = self.batch_size

        noise_batch = self.noise_shuffle[ self.batch_count- self.batch_size: self.batch_count]
        noise_batch = np.reshape(noise_batch, [self.batch_size, self.num_steps, self.plant_input_size])
        
        return noise_batch

    ## Step 1: generate data with real plant and current NN controller
    def train_data(self, episodes = 1000):
        print ('Testing controller...') 
        for episode in range(episodes):  
            ## noise input data
            curr_noise = self.gen_noise()
            curr_force = np.zeros([self.num_steps, self.plant_input_size])
            curr_resp = np.zeros([self.num_steps, self.plant_output_size])
            
            curr_pstate = self.gen_init_state()
            curr_cstate = self.zeros_one_cont_state
            curr_hstate = self.zeros_one_cont_state
            cont_out = 0.0
            for time in range(self.num_steps):
                curr_u = curr_noise[time] + cont_out
                oderes = si.odeint(self.model.dynamics, curr_pstate,self.t_arr,args=(curr_u,))
                curr_pstate = oderes[-1,:]
                if self.model_name == 'Pendulum':
                    resp_out = self.model.real_state_to_response(curr_pstate)
                else: pass

                cont_out, (curr_cstate, curr_hstate) = self.sess.run(self.cont_pred_tuple, feed_dict ={
                        self.cont_in: np.reshape(resp_out, [1,1,self.cont_input_size]), 
                        self.cstate_cont: curr_cstate, 
                        self.hstate_cont: curr_hstate})

                cont_out = np.reshape(cont_out, [ ])

                curr_force[time,:] = curr_u
                curr_resp[time,:] = resp_out
            
            self.add_fpn_data(curr_noise,curr_force,curr_resp)

    ## Step 2: train NN plant using measured data of real plant
    def train_plant(self, epochs = 10):
        print ('Training plant...') 
        self.shuffle_fp_data()
        self.batch_count=0
        for epo in range(epochs):
            print ('Epoch:{}'.format(epo + 1))
            
            for ind in range(math.floor(self.data_count/self.batch_size)):
                force_batch, resp_batch = self.get_fp_batch()

                self.sess.run(self.plant_optimizer, feed_dict = {self.plant_in: force_batch, 
                    self.cstate_plant: self.zeros_batch_plant_state, 
                    self.hstate_plant: self.zeros_batch_plant_state, 
                    self.plant_true: resp_batch})
            
            curr_cost = self.sess.run(self.plant_cost, feed_dict = {self.plant_in: force_batch, 
                self.cstate_plant: self.zeros_batch_cont_state, 
                self.hstate_plant: self.zeros_batch_cont_state, 
                self.plant_true : resp_batch})
            
            self.plant_hist.append(curr_cost)
            
        print ('Final train plant cost:{}'.format(curr_cost))

    ## Step 3: train NN controller using current NN plant
    def train_cont(self,epochs = 10):
        print ('Training controller...')
        train_steps = int(epochs*self.data_count/self.batch_size)
        self.shuffle_n_data()
        self.batch_count=0
        for epo in range(epochs):
            print ('Epoch:{}'.format(epo + 1))
            
            for ind in range(math.floor(self.data_count/self.batch_size)):
                noise_batch = self.get_n_batch()

                self.sess.run(self.fb_optimizer, feed_dict = {
                    self.fb_in: noise_batch, 
                    self.fb_diff: self.zeros_batch_plant_in,
                    self.cstate_plant: self.zeros_batch_plant_state, 
                    self.hstate_plant: self.zeros_batch_plant_state, 
                    self.cstate_cont: self.zeros_batch_cont_state, 
                    self.hstate_cont: self.zeros_batch_cont_state})
            
            curr_cost = self.sess.run(self.fb_cost, feed_dict = {
                self.fb_in: noise_batch, 
                self.fb_diff: self.zeros_batch_plant_in,
                self.cstate_plant: self.zeros_batch_plant_state, 
                self.hstate_plant: self.zeros_batch_plant_state, 
                self.cstate_cont: self.zeros_batch_cont_state, 
                self.hstate_cont: self.zeros_batch_cont_state})

            self.cont_hist.append(curr_cost)
            
        print ('Final train controller cost:{}'.format(curr_cost))

    def plot_plant_cost_hist(self):
        plt.plot(range(len(self.plant_hist)), self.plant_hist, '-', label = 'plant costs')
        plt.title('Plant cost history')
        plt.legend()        
        
    def plot_cont_cost_hist(self):    
        plt.plot(range(len(self.cont_hist)), self.cont_hist, '-', label = 'controller costs')
        plt.title('Controller cost history')
        plt.legend()      
        
    def plot_plant_dynamics(self, noise=None, init_state=None):

        if noise is None:
            curr_noise = self.gen_noise()
        else:
            curr_noise = noise
        if init_state is None:
            curr_pstate = self.gen_init_state()
        else:
            curr_pstate = init_state
        
        
        real_resp = np.zeros([self.num_steps, self.plant_output_size])
        for time in range(self.num_steps):
            curr_u = curr_noise[time]
            oderes = si.odeint(self.model.dynamics,curr_pstate,self.t_arr,args=(curr_u,))
            curr_pstate = oderes[-1,:]

            if self.model_name == 'Pendulum':
                resp_out = self.model.real_state_to_response(curr_pstate)

            real_resp[time,:] = resp_out

        ## Same noise data for plotting plant predicted position with/without controller
        pred_resp = self.sess.run(self.plant_pred, feed_dict = {
            self.plant_in: np.reshape(curr_noise, [1, self.num_steps, self.plant_input_size]), 
            self.cstate_plant: self.zeros_one_plant_state, 
            self.hstate_plant: self.zeros_one_plant_state})        
        pred_resp = np.reshape(pred_resp, [self.num_steps, self.plant_output_size])
        
        plot_time = np.arange(0, self.epitime, self.timestep)
        for ind in range(self.plant_output_size):
            plt.plot(plot_time, pred_resp[:,ind], '-', label = 'uncon pred ' + self.outvar_names[ind])
            plt.plot(plot_time, real_resp[:,ind], '-', label = 'uncon real ' + self.outvar_names[ind])
        plt.plot(plot_time, curr_noise, '-', label = 'input noise')
        plt.title('Uncontrolled plant response (Real vs Predicted)')
        plt.legend()

    def plot_cont_plant_dynamics(self,noise=None, init_state=None):  
        
        if noise is None:
            curr_noise = self.gen_noise()
        else:
            curr_noise = noise
        if init_state is None:
            curr_pstate = self.gen_init_state()
        else:
            curr_pstate = init_state
        
        real_resp = np.zeros([self.num_steps, self.plant_output_size])
        curr_pstate = self.gen_init_state()
        curr_cstate = self.zeros_one_cont_state
        curr_hstate = self.zeros_one_cont_state
        cont_out = 0.0
        for time in range(self.num_steps):
            curr_u = curr_noise[time] + cont_out
            oderes = si.odeint(self.model.dynamics,curr_pstate,self.t_arr,args=(curr_u,))
            curr_pstate = oderes[-1,:]
            if self.model_name == 'Pendulum':
                resp_out = self.model.real_state_to_response(curr_pstate)
            else: pass
            cont_out, (curr_cstate, curr_hstate) = self.sess.run(self.cont_pred_tuple, feed_dict ={
                    self.cont_in: np.reshape(resp_out, [1,1,self.cont_input_size]), 
                    self.cstate_cont: curr_cstate, 
                    self.hstate_cont: curr_hstate})
            cont_out = np.reshape(cont_out, [ ])
            real_resp[time,:] = resp_out
        
        ## Plot controlled signal with NN plant
        pred_resp = self.sess.run(self.fb_pred, feed_dict = {
            self.fb_in: np.reshape(curr_noise, [1, self.num_steps, self.plant_input_size]),
            self.fb_diff: self.zeros_one_plant_in,
            self.cstate_plant: self.zeros_one_plant_state, 
            self.hstate_plant: self.zeros_one_plant_state, 
            self.cstate_cont: self.zeros_one_cont_state, 
            self.hstate_cont: self.zeros_one_cont_state})
        pred_resp = np.reshape(pred_resp, [self.num_steps, self.plant_output_size])
            
        plot_time = np.arange(0, self.epitime, self.timestep)
        for ind in range(self.plant_output_size):
            plt.plot(plot_time, pred_resp[:,ind], '-', label = 'con pred ' + self.outvar_names[ind])
            plt.plot(plot_time, real_resp[:,ind], '-', label = 'con real ' + self.outvar_names[ind])
        plt.plot(plot_time, curr_noise, '-', label = 'input noise')
        plt.title('Controlled plant response (Real vs Predicted)')
        plt.legend()


## Main program
## An instance of controller class
LSTM_cont = LSTM_controller(epitime = 3.2, timestep= 0.1)
LSTM_cont.config_lstm(batch_size= 128, plant_state_size = 128, cont_state_size = 128, setout = [0.0,-1.0,0.0])
LSTM_cont.config_noise(nF = 5.0)
LSTM_cont.initialize_variables()
iterations = 2

LSTM_cont.train_data(episodes = 1024)
LSTM_cont.train_plant(epochs = 128)
LSTM_cont.train_cont(epochs = 32)

try:
    for it in range (iterations):
        print("Iteration:" + str(it+1))
        LSTM_cont.train_data(episodes = 32)
        LSTM_cont.train_plant(epochs = 16)
        LSTM_cont.train_cont(epochs = 16)

except KeyboardInterrupt:
    print("Execution interupted. Generating plots before ending.")

comp_noise = LSTM_cont.gen_noise()
comp_init_state = LSTM_cont.gen_init_state()
plt.figure(1)
LSTM_cont.plot_plant_dynamics(noise=comp_noise,init_state=comp_init_state)
plt.figure(2)
LSTM_cont.plot_cont_plant_dynamics(noise=comp_noise,init_state=comp_init_state)
plt.figure(3)
LSTM_cont.plot_plant_cost_hist()
plt.figure(4)
LSTM_cont.plot_cont_cost_hist()
plt.show()

print("Displaying plots")
print("Noise:")
print(comp_noise)
print("Initial state:")
print(comp_init_state)

LSTM_cont.sess.close()
