from __future__ import division
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
import numpy as np
import scipy.integrate as si
import random as random
import math
from copy import *
import pickle

import tensorflow as tf
import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'   # removes warning

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
        if name == 'Cavity_spring' : return Cavity_spring() 
        if name == 'Cartpole': return Cartpole()
        assert 0, 'Model name does not exist:' + name 

class Pendulum(Model):
    def __init__(self):
        self.config_dynamics()

    def config_dynamics(self, g=9.8, mu=0.1):
        self.g = g #pendumlem weight
        self.mu = mu #pendulem dampening

    def dynamics(self,state,t,u): 
        u = u[0]
        x, p = state
        dx = p
        dp =  -self.g * np.sin(x) - self.mu * p  + u
        return [dx,dp]

    def variables_name(self):
        return ["pos x","pos y","ang mom"]

    def real_state_to_response(self, real_state):
    ## Transforming angular position to x,y coordinate
        return np.array([18*np.sin(real_state[0]),-18*np.cos(real_state[0]),real_state[1]])

    def linear_control(self, state, t, u):
        u = u[0]
        x, p = state
        dx = p 
        dp =  -self.g * x - self.mu * p + u 
        return [dx,dp]

    def set_force_animation(self):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, aspect = 'equal', autoscale_on = False, xlim=(-0.2, 31), ylim = (-30, 30))
        self.ax.grid()
        self.line_noise_force, = self.ax.plot([],[],color='#FF4500' )
        self.line_cont_force, = self.ax.plot([],[], color='#1E90FF')
        self.line_lin_force, = self.ax.plot([],[],color='#008000' )
        self.noise_force_label = self.ax.text(0.6, 0.9, '', color='#FF4500', fontweight='bold', transform=self.ax.transAxes) 
        self.cont_force_label = self.ax.text(0.6, 0.85, '', color='#1E90FF', fontweight='bold', transform=self.ax.transAxes) 
        self.lin_force_label = self.ax.text(0.6, 0.8, '', color='#008000', fontweight='bold', transform=self.ax.transAxes)

    def init_force_animation(self):
        self.line_noise_force.set_data([],[])
        self.line_cont_force.set_data([],[])
        self.line_lin_force.set_data([],[])
        self.cont_force_label.set_text('')
        self.noise_force_label.set_text('')
        self.lin_force_label.set_text('')
        return self.line_noise_force, self.line_cont_force, self.line_lin_force, self.cont_force_label, self.noise_force_label, self.lin_force_label,

    def get_force_animation_data(self, noise_force, cont_force, lin_force, time_array):
        self.noise_force = noise_force
        self.cont_force = cont_force
        self.lin_force = lin_force
        self.time_array = time_array

    def force_animate(self, i):
        self.line_noise_force.set_data(self.time_array[0:i]*10, self.noise_force[0:i,0])
        self.line_cont_force.set_data(self.time_array[0:i]*10,self.cont_force[0:i,0])
        self.line_lin_force.set_data(self.time_array[0:i]*10, self.lin_force[0:i])
        self.noise_force_label.set_text('Noise')
        self.cont_force_label.set_text('NN')  
        self.lin_force_label.set_text('LQR')      
        return self.line_noise_force, self.line_cont_force, self.line_lin_force, self.noise_force_label, self.cont_force_label, self.lin_force_label,

    def show_force_animation(self, timestep, num_frames, file_name):
        self.set_force_animation()
        from time import time
        t0 = time()
        self.animate(0)
        t1 = time()
        interval = 1000 * timestep - (t1 - t0)
        ani = animation.FuncAnimation(self.fig, self.force_animate, frames= num_frames, 
            interval=interval, blit=True, init_func=self.init_animation)
        ani.save(file_name+'force.mp4', fps = int(1/timestep), extra_args=['-vcodec','libx264'])
        plt.show()

    def set_up_animation(self):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, aspect = 'equal', autoscale_on = False, xlim=(-60, 60), ylim = (-25, 25))
        self.ax.grid()
        self.line_cont, = self.ax.plot([],[],'o-',color='#1E90FF', lw = 2)
        self.line_real, = self.ax.plot([],[],'o-',color='#FF4500', lw = 2)
        self.line_lin, = self.ax.plot([],[],'o-', color='#008000', lw =2)
        self.time_text = self.ax.text(0.02,0.05, '', transform=self.ax.transAxes)
        self.cont_label = self.ax.text(0.4, 0.9, '', color='#1E90FF', fontweight='bold', transform=self.ax.transAxes) 
        self.real_label = self.ax.text(0.1, 0.9, '', color='#FF4500', fontweight='bold', transform=self.ax.transAxes)
        self.lin_label = self.ax.text(0.7, 0.9, '', color='#008000', fontweight='bold', transform=self.ax.transAxes)

    def init_animation(self):
        self.line_cont.set_data([],[])
        self.line_real.set_data([],[])
        self.line_lin.set_data([],[])
        self.time_text.set_text('')
        self.cont_label.set_text('')
        self.real_label.set_text('')
        self.lin_label.set_text('')
        return self.line_cont, self.line_real, self.line_lin, self.time_text, self.real_label, self.cont_label, self.lin_label,

    def get_animation_data(self, cont_data, uncont_data, lin_cont_data, time_array):
        self.cont_data = cont_data
        self.uncont_data = uncont_data
        self.lin_cont_data = lin_cont_data
        self.time_array = time_array

    def animate(self,i):
        self.line_cont.set_data([0, self.cont_data[:,0][i]], [0,self.cont_data[:,1][i]]) 
        self.line_real.set_data([-40,-40+self.uncont_data[:,0][i]], [0, self.uncont_data[:,1][i]])
        self.line_lin.set_data([40, 40+self.lin_cont_data[:,0][i]], [0, self.lin_cont_data[:,1][i]])
        self.time_text.set_text('Time=%.1f'%self.time_array[i])
        self.cont_label.set_text('NN Controlled')
        self.real_label.set_text('Uncontrolled')
        self.lin_label.set_text('LQR controlled')
        return self.line_cont, self.line_real, self.line_lin, self.time_text, self.real_label, self.cont_label, self.lin_label,

    def show_animation(self, timestep, num_frames, file_name):
        self.set_up_animation()
        from time import time
        t0 = time()
        self.animate(0)
        t1 = time()
        interval = 1000 * timestep - (t1 - t0)
        ani = animation.FuncAnimation(self.fig, self.animate, frames= num_frames, 
            interval=interval, blit=True, init_func=self.init_animation)
        ani.save(file_name+'.mp4', fps = int(1/timestep), extra_args=['-vcodec','libx264'])
        plt.show()

class Cavity_spring(Model):
    def __init__(self):
        self.config_dynamics()

    def config_dynamics(self, wc = 2.0, wm = 1.0, rc = 0.1, rm= 0.1, ks = 0.5):
        self.wc = wc # cavity oscillator frequency  
        self.wm = wm # mechanical oscillator frequency
        self.rc = rc # damping rate of cavity
        self.rm = rm # damping rate of mecahnical oscillator
        self.ks = ks # distance affecting frequency

    def dynamics(self, state, t, u):
        # uc: control of cavitiy, assume real for now
        # um: control of mechanical oscillator, assume real for now
        ar, ac, br, bc = state
        uc, um = u[0], u[1] 
        dar = uc -self.rc * ar + (self.wc - self.ks*br)*ac 
        dac = -self.rc*ac - (self.wc - self.ks*br)*ar 
        dbr = um - self.rm*br + self.wm*bc 
        dbc = -self.wm*br - self.rm*bc - self.ks *(ar*ar + ac*ac)
        return [dar, dac, dbr, dbc]

    def variables_name(self):
        return ['Re(alpha)', 'Im(alpha)', 'Re(beta)', 'Im(beta)']

class Cartpole(Model):
    def __init__(self):
        self.config_dynamics()

    def config_dynamics(self, mc = 0.5, mp = 0.2, muc = 0.1, l = 3 , i = 0.6, g = 9.8):
        self.mc = mc # mass of cart
        self.mp = mp # mass of pole
        self.muc = muc # coefficient of friction for cart
        self.l = l # length to of pendulum
        self.i = i # inertia of pendulum
        self.g = g 

    def dynamics(self, state, t, u):
        # u: control force applies to cart
        # x: cart position
        # theta: pendulum angle from vertical
        u= u[0]
        x, x_prime, theta, theta_prime, = state
        dx = x_prime
        dx_prime = (1/(self.mc+self.mp*np.sin(theta)*np.sin(theta)))*(u+self.mp*np.sin(theta)*(self.l*theta_prime*theta_prime-self.g*np.cos(theta)))
        dtheta = theta_prime
        dtheta_prime = (1/(self.l*(self.mc+self.mp*np.sin(theta)*np.sin(theta))))*(-u*np.cos(theta)-self.mp*self.l*theta_prime*theta_prime*np.sin(theta)*np.cos(theta)+(self.mp+self.mc)*self.g*np.sin(theta))

        return [dx, dx_prime, dtheta, dtheta_prime] 

    def variables_name(self):
        return ['pos x cart', 'cart velocity', 'x pos pole','y pos pole','pole ang mom'  ]

    def real_state_to_response(self, real_state):
    ## Transforming angular position to x,y coordinate
        # if -math.pi/2 <= real_state[0] <= math.pi/2:
        #     return np.array([np.sin(real_state[0])*self.l, -np.cos(real_state[0])*self.l ,real_state[1], real_state[2], real_state[3]])
        # else: 
            # return np.array([np.cos(real_state[0]-math.pi/2)*self.l, np.sin(real_state[0]-math.pi/2)*self.l ,real_state[1], real_state[2], real_state[3]]) 
        # return np.array([np.sin(real_state[0])*self.l, -np.cos(real_state[0])*self.l ,real_state[1], real_state[2], real_state[3]]) 
        return np.array([real_state[0], real_state[1], np.sin(real_state[2])*self.l, -np.cos(real_state[2])*self.l, real_state[3]])
    
    def set_force_animation(self):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, aspect = 'equal', autoscale_on = False, xlim=(-0.2, 31), ylim = (-30, 30))
        self.ax.grid()
        self.line_noise_force, = self.ax.plot([],[],color='#FF4500' )
        self.line_cont_force, = self.ax.plot([],[], color='#1E90FF')
        self.noise_force_label = self.ax.text(0.6, 0.9, '', color='#FF4500', fontweight='bold', transform=self.ax.transAxes) 
        self.cont_force_label = self.ax.text(0.6, 0.85, '', color='#1E90FF', fontweight='bold', transform=self.ax.transAxes) 

    def init_force_animation(self):
        self.line_noise_force.set_data([],[])
        self.line_cont_force.set_data([],[])
        self.cont_force_label.set_text('')
        self.noise_force_label.set_text('')
        return self.line_noise_force, self.line_cont_force, self.cont_force_label, self.noise_force_label,

    def get_force_animation_data(self, noise_force, cont_force, time_array):
        self.noise_force = noise_force
        self.cont_force = cont_force
        self.time_array = time_array

    def force_animate(self, i):
        self.line_noise_force.set_data(self.time_array[0:i]*10, self.noise_force[0:i,0])
        self.line_cont_force.set_data(self.time_array[0:i]*10,self.cont_force[0:i,0])
        self.noise_force_label.set_text('Noise')
        self.cont_force_label.set_text('Control')        
        return self.line_noise_force, self.line_cont_force, self.noise_force_label, self.cont_force_label,
    
    def show_force_animation(self, timestep, num_frames, file_name):
        self.set_force_animation()
        from time import time
        t0 = time()
        self.animate(0)
        t1 = time()
        interval = 1000 * timestep - (t1 - t0)
        ani = animation.FuncAnimation(self.fig, self.force_animate, frames= num_frames, 
            interval=interval, blit=True, init_func=self.init_animation)
        ani.save(file_name+'force.mp4', fps = int(1/timestep), extra_args=['-vcodec','libx264'])
        plt.show()        
    
    def set_up_animation(self):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, aspect='equal', autoscale_on = False, xlim=(-80, 80), ylim = (-30, 30))
        self.ax.grid()
        self.time_text = self.ax.text(0.02,0.05, '', transform=self.ax.transAxes)
        self.line_cont, = self.ax.plot([],[],'o-', lw = 2)
        self.line_real, = self.ax.plot([],[],'o-', lw = 2)
        self.patch_cont = patches.Rectangle((-12,-4), 12, 4, fc='b')
        self.patch_real = patches.Rectangle((-12,-4), 12, 4, fc='y')

    def init_animation(self):
        '''initialize animation'''
        self.ax.add_patch(self.patch_cont)
        self.ax.add_patch(self.patch_real)
        self.time_text.set_text('')
        self.line_cont.set_data([],[])
        self.line_real.set_data([],[])
        return self.line_cont, self.line_real, self.patch_cont, self.patch_real, self.time_text,

    def get_animation_data(self, cont_data, uncont_data, time_array):
        self.cont_data = cont_data
        self.uncont_data = uncont_data 
        self.time_array = time_array

    def animate(self,i):
        '''perform animation step'''
        # all positions are scaled by 8 for illustration
        self.line_cont.set_data([self.cont_data[:,0][i]*8, (self.cont_data[:,0][i]+self.cont_data[:,2][i])*8], [0, self.cont_data[:,3][i]*8])
        self.line_real.set_data([self.uncont_data[:,0][i]*8, (self.uncont_data[:,0][i]+self.uncont_data[:,2][i])*8], [0, self.uncont_data[:,3][i]*8])
        self.patch_cont.set_xy([self.cont_data[:,0][i]*8-6, -2])
        self.patch_real.set_xy([self.uncont_data[:,0][i]*8-6, -2])
        self.time_text.set_text('Time=%.1f'%self.time_array[i])
        return self.line_cont, self.line_real, self.patch_cont, self.patch_real,

    def show_animation(self, timestep, num_frames, model_name):
        self.set_up_animation()
        from time import time
        t0 = time()
        self.animate(0)
        t1 = time()
        interval = 1000 * timestep - (t1 - t0)
        ani = animation.FuncAnimation(self.fig, self.animate, frames= num_frames, 
            interval=interval, blit=True, init_func=self.init_animation)
        ani.save(model_name+'.mp4', fps = int(1/timestep), extra_args=['-vcodec','libx264'])
        plt.show()

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
        
    def config_lstm_para(self, batch_size = 128, plant_state_size = 128, cont_state_size = 32, 
                                input_size = 1, output_size = 3, learning_rate = 0.001, setout=None):
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

        # print ('setout:{}'.format(setout))
        ## call initialize states after all dimensions are defined    
        self.initialize_states()
    
    def initialize_states(self):
        # zero arrays 
        self.zeros_one_plant_state = np.zeros([1, self.plant_state_size])
        self.zeros_one_cont_state = np.zeros([1, self.cont_state_size])
        self.zeros_batch_plant_state = np.zeros([self.batch_size, self.plant_state_size])
        self.zeros_batch_cont_state = np.zeros([self.batch_size, self.cont_state_size])
        self.zeros_batch_plant_in = np.zeros([self.batch_size, self.plant_input_size])
        self.zeros_one_plant_in = np.zeros([1, self.plant_input_size])

        # Initialize plant_cstate, plant_hstate, cont_cstate, cont_hstate
        self.train_plant_cstate = self.zeros_batch_plant_state
        self.train_plant_hstate = self.zeros_batch_plant_state
        self.plot_plant_cstate = self.zeros_one_plant_state
        self.plot_plant_hstate = self.zeros_one_plant_state
        self.train_cont_cstate = self.zeros_batch_cont_state
        self.train_cont_hstate = self.zeros_batch_cont_state
        self.plot_cont_cstate = self.zeros_one_cont_state
        self.plot_cont_hstate = self.zeros_one_cont_state
        self.data_cont_cstate = self.zeros_one_cont_state
        self.data_cont_hstate = self.zeros_one_cont_state
        self.train_fb_diff = self.zeros_batch_plant_in
        self.plot_fb_diff = self.zeros_one_plant_in

    def config_lstm(self):
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

        # step 2 - fitting plant to data: cost and optimizer
        self.plant_cost = tf.reduce_mean((self.plant_pred - self.plant_true)*(self.plant_pred - self.plant_true))
        self.plant_optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.plant_cost, var_list=self.plant_var_list)
        
        # step 3 - fitting controller to plant and target: cost and optimizer
        if self.model_name == 'Cartpole':
            # for cartpole let x position of the cart be free
            self.fb_cost = tf.reduce_mean((self.fb_pred[:,:,1:5] - self.setout[1:5])*(self.fb_pred[:,:,1:5]-self.setout[1:5]))
        else: 
            self.fb_cost = tf.reduce_mean((self.fb_pred - self.setout)*(self.fb_pred-self.setout)) 
        ## Weighted cost function    
        self.fb_optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.fb_cost, var_list= self.cont_var_list)

    def config_noise(self, nF = [5], constant_noise = True):
        self.constant_noise = constant_noise # select noise type
        self.nF = [ ]
        for ind in range(len(nF)):
            self.nF.append(nF[ind]*math.sqrt(self.num_steps))  #magnitude of noise 
    
    def gen_noise(self):
        noise = np.zeros([self.num_steps, self.plant_input_size])
        if self.constant_noise: 
            for i in range(self.plant_input_size):
                temp=np.random.normal(0, self.nF[i])
                noise[:,i] = temp ## constant noise for all timesteps
        else:
            for i in range(self.plant_input_size):
                for time in range(self.num_steps):
                    noise[time,i] = np.random.normal(0,self.nF[i])
        return noise 

    def gen_init_state(self):
        if self.model_name == 'Pendulum':
            init_state = np.array([0.0, 0.0])
        elif self.model_name == 'Cartpole':
            init_state = np.array([0.0, 0.0, 0.0, 0.0])
        else: 
            init_state = np.array([0.0 for _ in range(self.plant_output_size)])
        
        return init_state
    
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
            curr_cstate = self.data_cont_cstate
            curr_hstate = self.data_cont_hstate
            
            curr_u = [0.0 for _ in range(self.plant_input_size)]
            cont_out = [0.0 for _ in range(self.plant_input_size)]

            for time in range(self.num_steps):
                for ind in range(self.plant_input_size):
                    curr_u[ind] = curr_noise[time][ind] + cont_out[ind]
                oderes = si.odeint(self.model.dynamics, curr_pstate, self.t_arr, args=(curr_u,))
                curr_pstate = oderes[-1,:]

                if self.model_name == 'Pendulum' or self.model_name == 'Cartpole':
                    resp_out = self.model.real_state_to_response(curr_pstate)
                    cont_out, (curr_cstate, curr_hstate) = self.sess.run(self.cont_pred_tuple, feed_dict ={
                        self.cont_in: np.reshape(resp_out, [1,1,self.cont_input_size]), 
                        self.cstate_cont: curr_cstate, 
                        self.hstate_cont: curr_hstate})
                    curr_resp[time,:] = resp_out
                else:
                    cont_out, (curr_cstate, curr_hstate) = self.sess.run(self.cont_pred_tuple, feed_dict ={
                            self.cont_in: np.reshape(curr_pstate, [1,1,self.cont_input_size]), 
                            self.cstate_cont: curr_cstate, 
                            self.hstate_cont: curr_hstate})
                    curr_resp[time,:] = curr_pstate

                cont_out = np.reshape(cont_out, [self.plant_input_size])
                curr_force[time,:] = curr_u

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
                    self.cstate_plant: self.train_plant_cstate,
                    self.hstate_plant: self.train_plant_hstate,
                    self.plant_true: resp_batch})
            
            curr_cost = self.sess.run(self.plant_cost, feed_dict = {self.plant_in: force_batch, 
                self.cstate_plant: self.train_plant_cstate,
                self.hstate_plant: self.train_plant_hstate,
                self.plant_true : resp_batch})
            
            self.plant_hist.append(curr_cost)

        print ('Final train plant cost:{}'.format(curr_cost))

    ## Step 3: train NN controller using current NN plant
    def train_cont(self,epochs = 10):
        print ('Training controller...')
        self.shuffle_n_data()
        self.batch_count=0
        for epo in range(epochs):
            print ('Epoch:{}'.format(epo + 1))
            
            for ind in range(math.floor(self.data_count/self.batch_size)):
                noise_batch = self.get_n_batch()

                self.sess.run(self.fb_optimizer, feed_dict = {
                    self.fb_in: noise_batch, 
                    self.fb_diff: self.train_fb_diff,
                    self.cstate_plant: self.train_plant_cstate,
                    self.hstate_plant: self.train_plant_hstate,
                    self.cstate_cont : self.train_cont_cstate,
                    self.hstate_cont : self.train_cont_hstate
                    })
            
            curr_cost = self.sess.run(self.fb_cost, feed_dict = {
                self.fb_in: noise_batch, 
                self.fb_diff: self.train_fb_diff,
                self.cstate_plant: self.train_plant_cstate,
                self.hstate_plant: self.train_plant_hstate,
                self.cstate_cont : self.train_cont_cstate,
                self.hstate_cont : self.train_cont_hstate
                })
            self.cont_hist.append(curr_cost)
        
        print ('Final train controller cost:{}'.format(curr_cost))

    '''Plotting methods'''
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
         
        self.real_uncont_resp = np.zeros([self.num_steps, self.plant_output_size])

        for time in range(self.num_steps):
            curr_u = curr_noise[time]
            oderes = si.odeint(self.model.dynamics,curr_pstate,self.t_arr,args=(curr_u,))
            curr_pstate = oderes[-1,:]

            if self.model_name == 'Pendulum' or self.model_name == 'Cartpole':
                reps_out = self.model.real_state_to_response(curr_pstate)
                self.real_uncont_resp[time,:] = reps_out

            else: 
                self.real_uncont_resp[time,:] = curr_pstate


        self.pred_resp = self.sess.run(self.plant_pred, feed_dict = {
            self.plant_in: np.reshape(curr_noise, [1, self.num_steps, self.plant_input_size]), 
            self.cstate_plant: self.plot_plant_cstate,
            self.hstate_plant: self.plot_plant_hstate
            })        
        self.pred_resp = np.reshape(self.pred_resp, [self.num_steps, self.plant_output_size])
        
        plot_time = np.arange(0, self.epitime, self.timestep)
        for ind in range(self.plant_output_size):
                plt.plot(plot_time, self.pred_resp[:,ind], '-', label = 'pred ' + self.outvar_names[ind])
                plt.plot(plot_time, self.real_uncont_resp[:,ind], '-', label = 'real ' + self.outvar_names[ind])

        plt.title('Uncontrolled system outputs (Real vs Predicted)')
        plt.xlabel('Time')
        plt.ylabel('System outputs')
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
        
        self.real_resp = np.zeros([self.num_steps, self.plant_output_size])
        curr_cstate = self.plot_cont_cstate
        curr_hstate = self.plot_cont_hstate

        curr_u = [0.0 for _ in range(self.plant_input_size)]
        cont_out = [0.0 for _ in range(self.plant_input_size)]
        self.cont_out_data = [ ]

        for time in range(self.num_steps):
            curr_u = curr_noise[time] + cont_out
            oderes = si.odeint(self.model.dynamics, curr_pstate, self.t_arr, args=(curr_u,))
            curr_pstate = oderes[-1,:]

            if self.model_name == 'Pendulum' or self.model_name == 'Cartpole':
                resp_out = self.model.real_state_to_response(curr_pstate)
                cont_out, (curr_cstate, curr_hstate) = self.sess.run(self.cont_pred_tuple, feed_dict ={
                    self.cont_in: np.reshape(resp_out, [1,1,self.cont_input_size]), 
                    self.cstate_cont: curr_cstate, 
                    self.hstate_cont: curr_hstate})
                self.real_resp[time,:] = resp_out
            else:
                cont_out, (curr_cstate, curr_hstate) = self.sess.run(self.cont_pred_tuple, feed_dict ={
                    self.cont_in: np.reshape(curr_pstate, [1,1,self.cont_input_size]), 
                    self.cstate_cont: curr_cstate, 
                    self.hstate_cont: curr_hstate})
                self.real_resp[time,:] = curr_pstate 
         
            cont_out = np.reshape(cont_out, [self.plant_input_size])
            
            self.cont_out_data.append(cont_out)

        ## Plot controlled signal with NN plant
        self.cont_pred_resp = self.sess.run(self.fb_pred, feed_dict = {
            self.fb_in: np.reshape(curr_noise, [1, self.num_steps, self.plant_input_size]),
            self.fb_diff: self.plot_fb_diff,
            self.cstate_plant: self.plot_plant_cstate,
            self.hstate_plant: self.plot_plant_hstate,
            self.cstate_cont: self.plot_cont_cstate,
            self.hstate_cont: self.plot_cont_hstate
            })
        self.cont_pred_resp = np.reshape(self.cont_pred_resp, [self.num_steps, self.plant_output_size])
            
        plot_time = np.arange(0, self.epitime, self.timestep)
        for ind in range(self.plant_output_size):
            plt.plot(plot_time, self.real_resp[:,ind], '-', label = 'cont real ' + self.outvar_names[ind])

        plt.title('Controlled system outputs')
        plt.xlabel('Time')
        plt.ylabel('System outputs')
        plt.legend()

    def plot_force(self, noise=None):
        self.curr_noise=noise
        plot_time = np.arange(0, self.epitime, self.timestep)
        self.cont_out_data = np.reshape(self.cont_out_data,[self.num_steps, self.plant_input_size])
        if self.model_name == 'Cavity_spring':
            var_names = ['cavity', 'oscillator']
            for i in range(self.plant_input_size):
                plt.plot(plot_time, self.cont_out_data[:,i], '-', label = 'cont force for '+var_names[i])
                plt.plot(plot_time, self.curr_noise[:,i], '-', label = 'input noise for '+var_names[i])
        else:
            plt.plot(plot_time, self.cont_out_data, '-', label = 'cont force')
            plt.plot(plot_time, self.curr_noise, '-', label = 'input noise')

        plt.title('Noise and control force')
        plt.xlabel('Time')
        plt.ylabel('Force (N)')
        plt.legend()

    ''' Animation methods: must call after calling plot_cont_plant_dynamics()'''
    def show_animation(self, filename):
        time_array = np.arange(0, self.epitime, self.timestep)
        if self.model_name == 'Pendulum':
            self.model.get_animation_data(self.real_resp, self.real_uncont_resp,self.linear_control_resp, time_array)
            self.model.get_force_animation_data(self.curr_noise, self.cont_out_data, self.linear_cont_data, time_array)
        else:
            self.model.get_animation_data(self.real_resp, self.real_uncont_resp, time_array)
            self.model.get_force_animation_data(self.curr_noise, self.cont_out_data,time_array)     
        self.model.show_animation(self.timestep, self.num_steps, filename)
        self.model.show_force_animation(self.timestep, self.num_steps, filename)

    ''' Saving/Loading and update '''
    def update_cont_plant_states(self):
        ''' allow last state values be fed into next iteration,
            If not called, state values for next iteration are zeros.'''
        self.train_fb_diff = self.sess.run(self.tf_state_tuple[-1], feed_dict ={
            self.fb_in: self.noise_data[-self.batch_size:], 
            self.fb_diff: self.train_fb_diff,
            self.cstate_plant: self.train_plant_cstate,
            self.hstate_plant: self.train_plant_hstate,
            self.cstate_cont : self.train_cont_cstate,
            self.hstate_cont : self.train_cont_hstate
            })

        (self.train_plant_cstate, self.train_plant_hstate) = self.sess.run(self.plant_state_tuple, feed_dict= {
            self.plant_in: self.force_data[-self.batch_size:], 
            self.cstate_plant: self.train_plant_cstate,
            self.hstate_plant: self.train_plant_hstate
            })

        cont_out, (self.data_cont_cstate, self.data_cont_hstate) = self.sess.run(self.cont_pred_tuple, feed_dict ={
            self.cont_in: np.reshape(self.pos_data[-1][-1:],[1,1,self.cont_input_size]),
            self.cstate_cont: self.data_cont_cstate, 
            self.hstate_cont: self.data_cont_hstate
            })
        resp = np.reshape(np.array(self.pos_data[-self.batch_size:])[:,-1,:] , [self.batch_size, 1, self.cont_input_size])
        cont_out, (self.train_cont_cstate, self.train_cont_hstate) = self.sess.run(self.cont_pred_tuple, feed_dict ={
            self.cont_in: resp,
            self.cstate_cont: self.train_cont_cstate, 
            self.hstate_cont: self.train_cont_hstate
            })
    
    def check_filename(self, string):
        return isinstance(string, str)

    def save_parameters(self, filename):
        ''' save parameters as a dictionary'''
        if self.check_filename(filename) :
            sim_dict = {'model_name':self.model_name, 'epitime': self.epitime, 'timestep': self.timestep, 'noise_magnitude': self.nF}
            para_dict ={'set_out': self.set_out, 'plant_state_size':self.plant_state_size, 
            'cont_state_size': self.cont_state_size, 'learning_rate': self.learning_rate }
            pickle.dump(sim_dict, open(filename+'sim_dict.p','wb')) 
            pickle.dump(para_dict,open(filename+'para_dict.p','wb'))
        else: assert 0, 'Enter filename as a string.'

    def save_data(self,filename):
        ''' save all training data'''
        if self.check_filename(filename) : 
            pickle.dump(noise_data, open(filename+'noise_data.p','wb'))
            pickle.dump(force_data, open(filename+'force_data.p','wb'))
            pickle.dump(pos_data, open(filename+'pos_data.p','wb'))
        else: assert 0, 'Enter filename as a string.'

    def save_trained_NN(self, filename):
        if self.check_filename(filename):
            my_vars = [self.plant_in, self.plant_optimizer, self.plant_cost,  self.cstate_plant, self.hstate_plant, self.plant_true, self.plant_pred,
                self.cont_pred_tuple[0], self.cont_pred_tuple[1][0], self.cont_pred_tuple[1][1], self.cont_in, self.cstate_cont, 
                 self.hstate_cont, self.fb_optimizer, self.fb_cost, self.fb_in, self.fb_diff, self.fb_pred]
            saver = tf.train.Saver()
            for v in my_vars:
                tf.add_to_collection(filename+'NN_var', v)
                saver.save(self.sess, filename+'NN_save')
            for v in self.plant_var_list:
                tf.add_to_collection(filename+'plant_var', v)
                saver.save(self.sess, filename+'plant_save')
            for v in self.cont_var_list:
                tf.add_to_collection(filename+'cont_var',v)
                saver.save(self.sess, filename+'cont_save')
        else: assert 0, 'Enter filename as a string.'

    def load_parameters(self, filename):
        if self.check_filename(filename):
            sim_dict = pickle.load(open(filename+'sim_dict.p','rb'))
            para_dict = pickle.load(open(filename+'para_dict.p','rb'))
            model_name, epitime, timestep, noise_magnitude = sim_dict['model_name'], sim_dict['epitime'], sim_dict['timestep'], sim_dict['noise_magnitude']
            set_out, plant_state_size, cont_state_size, learning_rate = para_dict['set_out'], para_dict['plant_state_size'], para_dict['cont_state_size'], para_dict['learning_rate']
            return model_name, epitime, timestep, noise_magnitude, set_out, plant_state_size, cont_state_size, learning_rate
        else: assert 0, 'Enter filename as a string.'

    def load_data(self, filename):
        if self.check_filename(filename):
            noise_data = pickle.load(filename+'noise_data.p','rb')
            force_data = pickle.load(filename+'force_data.p','rb')
            pos_data = pickle.load(filename+'pos_data.p', 'rb')
            return noise_data, force_data, pos_data
        else: assert 0, 'Enter filename as a string.'

    def load_trained_NN(self, filename):
        if self.check_filename(filename):
            saver = tf.train.import_meta_graph(filename+'NN_save.meta')
            saver.restore(self.sess, filename+'NN_save')
            [self.plant_in, self.plant_optimizer, self.plant_cost, self.cstate_plant, self.hstate_plant, self.plant_true, self.plant_pred,
                self.cont_pred, self.cont_cstate, self.cont_hstate, self.cont_in, self.cstate_cont,  self.hstate_cont, self.fb_optimizer, 
                self.fb_cost, self.fb_in, self.fb_diff,self.fb_pred]= tf.get_collection(filename+'NN_var')

            self.cont_pred_tuple = self.cont_pred, (self.cont_cstate, self.cont_hstate)
            
            plant_saver = tf.train.import_meta_graph(filename+'plant_save.meta')
            plant_saver.restore(self.sess, filename+'plant_save')
            self.plant_var_list = tf.get_collection(filename+'plant_var')
            cont_saver = tf.train.import_meta_graph(filename+'cont_save.meta')
            cont_saver.restore(self.sess, filename+'cont_save')
            self.cont_var_list = tf.get_collection(filename+'cont_var')
        else: assert 0, 'Enter filename as a string.'

    ## Testing controller performance
    def test_controller(self,num_test_samples, setpoint):
        pos_place = tf.placeholder(self.dtype, [None, self.num_steps, self.plant_output_size])
        test_costfunc= tf.reduce_mean((pos_place - self.setout)*(pos_place - self.setout))
        test_cost = self.sess.run(test_costfunc, feed_dict={pos_place: self.pos_data}) 
        print ('controller test cost:{}'.format(test_cost))

    def test_systemid(self, num_test_samples):
        test_cost = self.sess.run(self.plant_cost, feed_dict={self.plant_in: self.force_data,
            self.cstate_plant: np.zeros([num_test_samples, self.plant_state_size]),
            self.hstate_plant: np.zeros([num_test_samples, self.plant_state_size]),
            self.plant_true: self.pos_data})
        print ('plant test cost:{}'.format(test_cost))

    def test_linear_control(self, num_test_samples,init_state=None):
        curr_pstate=init_state
        linear_control_resp = np.zeros([num_test_samples,self.num_steps,self.plant_output_size])
        for i in range(num_test_samples):
            curr_noise= np.reshape(self.noise_data[i,:,:],[self.num_steps])
            for time in range(self.num_steps):
                curr_u = curr_noise[time] 
                oderes = si.odeint(self.model.linear_control, curr_pstate, self.t_arr, args=(curr_u,))
                curr_pstate = oderes[-1,:]

                if self.model_name == 'Pendulum' or self.model_name == 'Cartpole':
                    resp_out = self.model.real_state_to_response(curr_pstate)
                    linear_control_resp[i,time,:] = resp_out
        linear_control_place=tf.placeholder(self.dtype, [None, self.num_steps, self.plant_output_size])  
        test_costfunc= tf.reduce_mean((linear_control_place - self.setout)*(linear_control_place - self.setout))  
        test_cost = self.sess.run(test_costfunc, feed_dict={linear_control_place: linear_control_resp})       
        print('lqr test cost:{}'.format(test_cost))

    def plot_linear_control(self, noise=None,init_state=None):
        ## this is only for the pendulum and cartpole
        ## control force is 1D
        curr_pstate=init_state
        self.linear_control_resp = np.zeros([self.num_steps,self.plant_output_size])
        self.linear_cont_data = [0]
        linear_cont_force = 0
        curr_noise=noise
        plot_time = np.arange(0, self.epitime, self.timestep)
        for time in range(self.num_steps):
            curr_u = curr_noise[time] + linear_cont_force
            oderes = si.odeint(self.model.dynamics, curr_pstate, self.t_arr, args=(curr_u,))
            curr_pstate = oderes[-1,:]
            linear_cont_force = -90*(curr_pstate[0]-0) - 8*curr_pstate[1]
            self.linear_cont_data.append(linear_cont_force)
            if self.model_name == 'Pendulum' or self.model_name == 'Cartpole':
                resp_out = self.model.real_state_to_response(curr_pstate)
                self.linear_control_resp[time,:] = resp_out
        self.linear_cont_data = np.array(self.linear_cont_data)
        for ind in range(self.plant_output_size):
            plt.plot(plot_time, self.linear_control_resp[:,ind], '-', label = 'linear cont ' + self.outvar_names[ind])
        plt.title('Linear controlled system outputs')
        plt.xlabel('Time')
        plt.ylabel('System outputs')
        plt.legend()