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
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'   # removes warning

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

    def config_dynamics(self, g=9.8, mu=0.3):
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

    def set_up_animation(self):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, aspect = 'equal', autoscale_on = True, xlim=(-10, 10), ylim = (-2, 2))
        self.ax.grid()
        self.line_cont, = self.ax.plot([],[],'o-', lw = 2)
        self.line_real, = self.ax.plot([],[],'o-', lw = 2)

    def init_animation(self):
        self.line_cont.set_data([],[])
        self.line_real.set_data([],[])
        return self.line_cont, self.line_real,

    def get_animation_data(self, cont_data, uncont_data):
        self.cont_data = cont_data
        self.uncont_data = uncont_data

    def animate(self,i):
        self.line_cont.set_data([0, self.cont_data[:,0][i]], [0,self.cont_data[:,1][i]]) 
        self.line_real.set_data([0, self.uncont_data[:,0][i]], [0, self.uncont_data[:,1][i]])
        return self.line_cont, self.line_real,

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

class Cavity_spring(Model):
    def __init__(self):
        self.config_dynamics()

    def config_dynamics(self, wc = 1.0, wm = 1.0, rc = 0.1, rm= 0.1, ks = 0.1):
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
        dar = uc -self.rc * ar + (self.wc - self.ks)*br*ac 
        dac = -self.rc*ac - (self.wc - self.ks)*br*ar 
        dbr = um - self.rm*br + self.wm*bc 
        dbc = -self.wm*br - self.rm*bc - self.ks *(ar*ar + ac*ac)
        return [dar, dac, dbr, dbc]

    def variables_name(self):
        return ['real a', 'complex a', 'real b', 'complex b']

class Cartpole(Model):
    def __init__(self):
        self.config_dynamics()

    def config_dynamics(self, mc = 0.5, mp = 0.2, muc = 0.1, l = 0.3 , i = 0.006, g = 9.8):
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
        x, x_prime, theta, theta_prime, = state
        dx = x_prime
        dx_prime = (1/(self.mc+self.mp))* (u -self.muc*x_prime - self.mp*self.l*(np.cos(theta)+theta_prime*theta_prime*np.sin(theta)))
        dtheta = theta_prime
        dtheta_prime = (1/(self.mp*self.l*self.l))*(-self.mp* self.g*self.l*np.sin(theta) - self.mp*self.l*dx_prime *np.cos(theta))

        return [dx, dx_prime, dtheta, dtheta_prime] 

    def variables_name(self):
        return ['cart x', 'cart velocity', 'pole angle','pole ang velocity'  ]

    def real_state_to_response(self, real_state):
    ## Transforming angular position to x,y coordinate
        # if -math.pi/2 <= real_state[0] <= math.pi/2:
        #     return np.array([np.sin(real_state[0])*self.l, -np.cos(real_state[0])*self.l ,real_state[1], real_state[2], real_state[3]])
        # else: 
            # return np.array([np.cos(real_state[0]-math.pi/2)*self.l, np.sin(real_state[0]-math.pi/2)*self.l ,real_state[1], real_state[2], real_state[3]]) 
        return np.array([np.sin(real_state[0])*self.l, -np.cos(real_state[0])*self.l ,real_state[1], real_state[2], real_state[3]]) 


    def set_up_animation(self):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, aspect='equal', autoscale_on = True, xlim=(-20, 20), ylim = (-3, 3))
        self.ax.grid()
        self.line_cont, = self.ax.plot([],[],'o-', lw = 1)
        self.line_real, = self.ax.plot([],[],'o-', lw = 1)
        self.patch_cont = patches.Rectangle((-1,-0.5), 1, 0.5, fc='b')
        self.patch_real = patches.Rectangle((-1,-0.5), 1, 0.5, fc='y')

    def init_animation(self):
        '''initialize animation'''
        self.ax.add_patch(self.patch_cont)
        self.ax.add_patch(self.patch_real)
        self.line_cont.set_data([],[])
        self.line_real.set_data([],[])
        return self.line_cont, self.line_real, self.patch_cont, self.patch_real, 

    def get_animation_data(self, cont_data, uncont_data):
        self.cont_data = cont_data
        self.uncont_data = uncont_data 

    def animate(self,i):
        '''perform animation step'''
        self.line_cont.set_data([self.cont_data[:,3][i], self.cont_data[:,3][i]+self.cont_data[:,0][i]], [-0.25, self.cont_data[:,1][i]])
        self.line_real.set_data([self.uncont_data[:,3][i], self.uncont_data[:,3][i]+self.uncont_data[:,0][i]], [-0.25, self.uncont_data[:,1][i]])
        self.patch_cont.set_xy([self.cont_data[:,3][i]-0.5, -0.5])
        self.patch_real.set_xy([self.uncont_data[:,3][i]-0.5, -0.5])
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

        print ('setout:{}'.format(setout))
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
    
    ## TODO: potentially create multiple layers LSTM
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
        self.fb_cost = tf.reduce_mean((self.fb_pred - self.setout)*(self.fb_pred - self.setout))
        # self.fb_cost = tf.reduce_mean(self.fb_pred*self.fb_pred)
        self.fb_optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.fb_cost, var_list= self.cont_var_list)

    def config_noise(self, nF = 5, constant_noise = True):
        self.constant_noise = constant_noise # select noise type
        self.nF = nF #magnitude of noise 
    
    def gen_noise(self):
        if self.constant_noise: 
            temp = np.random.normal(0, self.nF)
            noise = np.full([self.num_steps, self.plant_input_size], temp) ## constant noise for all timesteps
        else:
            noise = [np.full([self.plant_input_size], np.random.normal(0, self.nF)) for _ in range(self.num_steps)]
        return noise 
        # return self.nF*np.random.randn([self.num_steps, self.plant_input_size])
    
    def gen_init_state(self):
        if self.model_name == 'Pendulum':
            init_state = np.array([0.0, 0.0])
        elif self.model_name == 'Cartpole':
            init_state = np.array([0.0, 0.0, 0.0, 0.0])
        else: 
            init_state = np.array([0.0 for _ in range(self.plant_output_size)])
        
        return init_state
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
            # curr_cstate = self.zeros_one_cont_state
            # curr_hstate = self.zeros_one_cont_state
            curr_cstate = self.data_cont_cstate
            curr_hstate = self.data_cont_hstate
            
            curr_u = [0.0 for _ in range(self.plant_input_size)]
            cont_out = [0.0 for _ in range(self.plant_input_size)]

            for time in range(self.num_steps):
                for ind in range(self.plant_input_size):
                    curr_u[ind] = curr_noise[time][ind] + cont_out[ind]
                # curr_u = curr_noise[time] + cont_out
                oderes = si.odeint(self.model.dynamics, curr_pstate, self.t_arr, args=(curr_u,))
                curr_pstate = oderes[-1,:]

                # if self.model_name == 'Pendulum' or 'Cartpole':
                if self.model_name == 'Pendulum':
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
            print (self.force_data)

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
                    # self.cstate_plant: self.zeros_batch_plant_state, 
                    # self.hstate_plant: self.zeros_batch_plant_state, 
                    self.cstate_plant: self.train_plant_cstate,
                    self.hstate_plant: self.train_plant_hstate,
                    self.plant_true: resp_batch})
            
            curr_cost = self.sess.run(self.plant_cost, feed_dict = {self.plant_in: force_batch, 
                # self.cstate_plant: self.zeros_batch_cont_state, 
                # self.hstate_plant: self.zeros_batch_cont_state, 
                self.cstate_plant: self.train_plant_cstate,
                self.hstate_plant: self.train_plant_hstate,
                self.plant_true : resp_batch})
            
            self.plant_hist.append(curr_cost)

        print ('Final train plant cost:{}'.format(curr_cost))


    ## Step 3: train NN controller using current NN plant
    def train_cont(self,epochs = 10):
        print ('Training controller...')
        # train_steps = int(epochs*self.data_count/self.batch_size)
        self.shuffle_n_data()
        self.batch_count=0
        for epo in range(epochs):
            print ('Epoch:{}'.format(epo + 1))
            
            for ind in range(math.floor(self.data_count/self.batch_size)):
                noise_batch = self.get_n_batch()

                self.sess.run(self.fb_optimizer, feed_dict = {
                    self.fb_in: noise_batch, 
                    # self.fb_diff: self.zeros_batch_plant_in,
                    self.fb_diff: self.train_fb_diff,
                    # self.cstate_plant: self.zeros_batch_plant_state, 
                    # self.hstate_plant: self.zeros_batch_plant_state,
                    self.cstate_plant: self.train_plant_cstate,
                    self.hstate_plant: self.train_plant_hstate,
                    # self.cstate_cont: self.zeros_batch_cont_state, 
                    # self.hstate_cont: self.zeros_batch_cont_state
                    self.cstate_cont : self.train_cont_cstate,
                    self.hstate_cont : self.train_cont_hstate
                    })
            
            curr_cost = self.sess.run(self.fb_cost, feed_dict = {
                self.fb_in: noise_batch, 
                # self.fb_diff: self.zeros_batch_plant_in,
                self.fb_diff: self.train_fb_diff,
                # self.cstate_plant: self.zeros_batch_plant_state, 
                # self.hstate_plant: self.zeros_batch_plant_state, 
                self.cstate_plant: self.train_plant_cstate,
                self.hstate_plant: self.train_plant_hstate,
                # self.cstate_cont: self.zeros_batch_cont_state, 
                # self.hstate_cont: self.zeros_batch_cont_state
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
        
    def plot_plant_dynamics(self, noise=None, init_state=None, update_plot_state = False):

        if noise is None:
            curr_noise = self.gen_noise()
        else:
            curr_noise = noise
        if init_state is None:
            curr_pstate = self.gen_init_state()
        else:
            curr_pstate = init_state
         
        if self.model_name == 'Cartpole':
            self.real_uncont_resp = np.zeros([self.num_steps, self.plant_output_size+1])
            real_uncont_angle = np.zeros([self.num_steps, self.plant_output_size])
        else:
            self.real_uncont_resp = np.zeros([self.num_steps, self.plant_output_size])


        for time in range(self.num_steps):
            curr_u = curr_noise[time]
            oderes = si.odeint(self.model.dynamics,curr_pstate,self.t_arr,args=(curr_u,))
            curr_pstate = oderes[-1,:]

            if self.model_name == 'Cartpole':
                reps_out = self.model.real_state_to_response(curr_pstate)
                self.real_uncont_resp[time,:] = reps_out
                real_uncont_angle[time,:] = curr_pstate

            elif self.model_name == 'Pendulum':
                reps_out = self.model.real_state_to_response(curr_pstate)
                self.real_uncont_resp[time,:] = reps_out

            else: self.real_uncont_resp[time,:] = reps_out


        pred_resp = self.sess.run(self.plant_pred, feed_dict = {
            self.plant_in: np.reshape(curr_noise, [1, self.num_steps, self.plant_input_size]), 
            # self.cstate_plant: self.zeros_one_plant_state, 
            # self.hstate_plant: self.zeros_one_plant_state
            self.cstate_plant: self.plot_plant_cstate,
            self.hstate_plant: self.plot_plant_hstate
            })        
        pred_resp = np.reshape(pred_resp, [self.num_steps, self.plant_output_size])
        
        plot_time = np.arange(0, self.epitime, self.timestep)
        for ind in range(self.plant_output_size):
                plt.plot(plot_time, pred_resp[:,ind], '-', label = 'uncon pred ' + self.outvar_names[ind])
                if self.model_name == 'Cartpole':
                    plt.plot(plot_time, real_uncont_angle[:,ind], '-', label = 'uncon real ' + self.outvar_names[ind])
                else: plt.plot(plot_time, self.real_uncont_resp[:,ind], '-', label = 'uncon real ' + self.outvar_names[ind])

        plt.plot(plot_time, curr_noise, '-', label = 'input noise')
        plt.title('Uncontrolled plant response (Real vs Predicted)')
        plt.legend()

        # update next plot plant states
        if update_plot_state:   
            self.plot_plant_cstate, self.plot_plant_hstate = self.sess.run(self.plant_state_tuple, feed_dict= {
                self.plant_in: np.reshape(curr_noise, [1, self.num_steps, self.plant_input_size]),
                self.cstate_plant: self.plot_plant_cstate,
                self.hstate_plant: self.plot_plant_hstate
                })
        else: pass 

    def plot_cont_plant_dynamics(self,noise=None, init_state=None, update_plot_state = False):  
        
        if noise is None:
            curr_noise = self.gen_noise()
        else:
            curr_noise = noise
        if init_state is None:
            curr_pstate = self.gen_init_state()
        else:
            curr_pstate = init_state
        
        if self.model_name == 'Cartpole':
            self.real_resp = np.zeros([self.num_steps, self.plant_output_size+1])
            real_resp = np.zeros([self.num_steps, self.plant_output_size])
        else: 
            self.real_resp = np.zeros([self.num_steps, self.plant_output_size])
        # curr_cstate = self.zeros_one_cont_state
        # curr_hstate = self.zeros_one_cont_state
        curr_cstate = self.plot_cont_cstate
        curr_hstate = self.plot_cont_hstate

        curr_u = [0.0 for _ in range(self.plant_input_size)]
        cont_out = [0.0 for _ in range(self.plant_input_size)]

        for time in range(self.num_steps):
            curr_u = curr_noise[time] + cont_out
            oderes = si.odeint(self.model.dynamics, curr_pstate, self.t_arr, args=(curr_u,))
            curr_pstate = oderes[-1,:]


            if self.model_name == 'Pendulum':
                resp_out = self.model.real_state_to_response(curr_pstate)
                cont_out, (curr_cstate, curr_hstate) = self.sess.run(self.cont_pred_tuple, feed_dict ={
                    self.cont_in: np.reshape(resp_out, [1,1,self.cont_input_size]), 
                    self.cstate_cont: curr_cstate, 
                    self.hstate_cont: curr_hstate})
                self.real_resp[time,:] = resp_out

            elif self.model_name == 'Cartpole':
                cont_out, (curr_cstate, curr_hstate) = self.sess.run(self.cont_pred_tuple, feed_dict ={
                self.cont_in: np.reshape(curr_pstate, [1,1,self.cont_input_size]), 
                self.cstate_cont: curr_cstate, 
                self.hstate_cont: curr_hstate})
                real_resp[time,:] = curr_pstate
                
                resp_out = self.model.real_state_to_response(curr_pstate)
                self.real_resp[time,:] = resp_out

            else:
                cont_out, (curr_cstate, curr_hstate) = self.sess.run(self.cont_pred_tuple, feed_dict ={
                    self.cont_in: np.reshape(curr_pstate, [1,1,self.cont_input_size]), 
                    self.cstate_cont: curr_cstate, 
                    self.hstate_cont: curr_hstate})
                self.real_resp[time,:] = curr_pstate
         
            cont_out = np.reshape(cont_out, [self.plant_input_size])
    
        ## Plot controlled signal with NN plant
        pred_resp = self.sess.run(self.fb_pred, feed_dict = {
            self.fb_in: np.reshape(curr_noise, [1, self.num_steps, self.plant_input_size]),
            # self.fb_diff: self.zeros_one_plant_in,
            self.fb_diff: self.plot_fb_diff,
            # self.cstate_plant: self.zeros_one_plant_state, 
            # self.hstate_plant: self.zeros_one_plant_state, 
            self.cstate_plant: self.plot_plant_cstate,
            self.hstate_plant: self.plot_plant_hstate,
            # self.cstate_cont: self.zeros_one_cont_state, 
            # self.hstate_cont: self.zeros_one_cont_state
            self.cstate_cont: self.plot_cont_cstate,
            self.hstate_cont: self.plot_cont_hstate
            })
        pred_resp = np.reshape(pred_resp, [self.num_steps, self.plant_output_size])
            
        plot_time = np.arange(0, self.epitime, self.timestep)
        for ind in range(self.plant_output_size):
            plt.plot(plot_time, pred_resp[:,ind], '-', label = 'con pred ' + self.outvar_names[ind])
            if self.model_name == 'Cartpole':
                plt.plot(plot_time, real_resp[:,ind], '-', label = 'con real ' + self.outvar_names[ind])
            else: 
                plt.plot(plot_time, self.real_resp[:,ind], '-', label = 'con real ' + self.outvar_names[ind])
        plt.plot(plot_time, curr_noise, '-', label = 'input noise')
        plt.title('Controlled plant response (Real vs Predicted)')
        plt.legend()

        # update next state values
        if update_plot_state :
            self.plot_cont_cstate, self.plot_cont_hstate = curr_cstate, curr_hstate

            self.plot_fb_diff = self.sess.run(self.tf_state_tuple[-1], feed_dict={
                self.fb_in:  np.reshape(curr_noise, [1, self.num_steps, self.plant_input_size]),
                self.fb_diff: self.plot_fb_diff, 
                self.cstate_plant: self.plot_plant_cstate,
                self.hstate_plant: self.plot_plant_hstate,
                self.cstate_cont: self.plot_cont_cstate,
                self.hstate_cont: self.plot_cont_hstate
                })
        else: pass

    ''' Animation methods: must call after calling plot_cont_plant_dynamics()'''
    def show_animation(self):
        self.model.get_animation_data(self.real_resp, self.real_uncont_resp)
        self.model.show_animation(self.timestep, self.num_steps, self.model_name)

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
                 self.hstate_cont, self.fb_optimizer, self.fb_cost, self.fb_in, self.fb_diff]
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
                self.fb_cost, self.fb_in, self.fb_diff]= tf.get_collection(filename+'NN_var')

            self.cont_pred_tuple = self.cont_pred, (self.cont_cstate, self.cont_hstate)
            
            plant_saver = tf.train.import_meta_graph(filename+'plant_save.meta')
            plant_saver.restore(self.sess, filename+'plant_save')
            self.plant_var_list = tf.get_collection(filename+'plant_var')
            cont_saver = tf.train.import_meta_graph(filename+'cont_save.meta')
            cont_saver.restore(self.sess, filename+'cont_save')
            self.cont_var_list = tf.get_collection(filename+'cont_var')
        else: assert 0, 'Enter filename as a string.'
