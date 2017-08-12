from __future__ import division
import numpy as np
from scipy.integrate import odeint
import random as random
import math as math
from copy import *
import pickle

NULL = ''
'''
Damped Pendulum equations of motion
# y1 = angular position
# y2 = angular velocity
'''
def nonlinear_damped_pen_position (y, t, g, mu, f ):
    y1,y2 = y
    dydt = [y2, -g*np.sin(y1) - mu*y2  + f ]
    return dydt

def linear_damped_pen_position (y, t, g, mu, f):
	y1, y2 = y 
	dydt = [y2, -g*y1 - mu*y2 + f]
	return dydt

'''
Driven damped pendulum simulation class
Simulated results for a time series of ngular position and velocity given a time series of random input force

Input args: 
## Pendulum parameters: g = 9.8 , mu = frictional force, , y_int = initial angular position, v_int = initial angular velocity, 
f = bound of magnitude of input random force
## Simulation run time: maxtime = maximum running time, timestep

Outputs: 
step_nums = int(ceil(maxtime/timestep)) 
F = list of random force picked from normal distribution with mean = 0 and sigma = f * sqrt(steps_num)
Y_non = list of angular position using the nonlinear model
V_non = list of angular velocity using the nonlinear model
Y_lin = list of angular position using the linear model
V_lin = list of angular velocity using the linear model
'''
class pendulum(object):
	def __init__(self):
		self.set_variables()
		self.init_variables()
		self.zero_arrays()

	def is_valid(self,a):
		return isinstance(a,int) or isinstance(a,float)

	def set_variables(self, g = NULL, mu = NULL, f = NULL, y_int = NULL, v_int = NULL):
		if self.is_valid(g ): self.g  = g
		if self.is_valid(mu): self.mu = mu
		if self.is_valid(f ): self.f  = f 
		if self.is_valid(y_int): self.y_int = y_int
		if self.is_valid(v_int): self.v_int = v_int 

	def init_variables(self):
		self.set_variables( g = 9.8, mu = 0.1, f = 5, y_int = 0, v_int = 0)

	def zero_arrays(self):
		self.F = [  ]
		self.Y_non = [self.y_int]
		self.V_non = [self.v_int]
		self.Y_lin = [self.y_int]
		self.V_lin = [self.v_int]

	def steps_num(self, maxtime, timestep):
		return int(math.ceil(maxtime/timestep))

	def input_force(self, steps_num):
		mean = 0
		sigma = self.f * math.sqrt(steps_num)
		force = np.random.normal(mean, sigma) 
		return force

	def nonlinear_update(self, timestep, force):
		t = [0 ,timestep]
		yv_sol = odeint(nonlinear_damped_pen_position,
						[ self.Y_non[-1], self.V_non[-1] ],
						t,
						args = (self.g, self.mu, force))
		y_next, v_next = yv_sol[-1]
		self.Y_non.append(y_next)
		self.V_non.append(v_next)

	def linear_update(self, timestep, force):
	    t=[0, timestep]
	    yv_sol = odeint (linear_damped_pen_position,
	                    [ self.Y_lin[-1], self.V_lin[-1] ], 
	                    t, 
	                    args= (self.g, self.mu, force))
	    y_next, v_next = yv_sol[-1]  
	    self.Y_lin.append(y_next)
	    self.V_lin.append(v_next)

	def run(self, maxtime = 100, timestep = 0.1):
		num_steps = self.steps_num(maxtime, timestep)
		for _ in range(num_steps):
			force = self.input_force(num_steps)
			self.F.append(force)
			self.nonlinear_update(timestep ,force)
			self.linear_update(timestep ,force)
		##### start sampling at time step 1, don't care about initial point
		self.Y_lin = self.Y_lin[1:]
		self.Y_non = self.Y_non[1:]
		self.V_lin = self.V_lin[1:]
		self.V_non = self.V_non[1:]

	def results(self):
		pendulum_dict = {"Y_non": self.Y_non , "V_non": self.V_non, "Y_lin": self.Y_lin, "V_lin": self.V_lin, "F": self.F}
		return pendulum_dict


'''
PID class 
Simulated results for PID controller for controlling a damped pendulum 
'''

## Define the integral approximations 
def integrate_sum_rectangle(E, ts):
    return sum(E)*ts

def integrate_sum_trapezoid(E, ts):
    traps = [ts*0.5*(E[i]+E[i-1]) for i in range(1,len(E))]
    return sum(traps) 

def integrate_sum_simpson(E, ts):
    quads = [(timestep/3)*(E[i-1] + 4*E[i] + E[i+1]) for i in xrange(1,len(E)-1)]
    return sum(quads) 	

## PID class
class PID_delay(object):
	Approximations = ["euler", "backwards"]
	Integrals      = ["rectangle", "trapezoid", "simpson"]
	## Initialize a class with given maxtime and timestep
	def __init__(self):
		self.init_variables()
		self.zero_arrays()
		self.set_approximation()
		self.set_integral()	

	def is_valid(self, a):
		return isinstance(a, int) or isinstance(a, float)

	def init_variables(self):
		self.set_variables(S = 1, kp = 1.0, ki = 8.0, kd = 15.0, mu = 0.1, g= 9.8)

	## Set variables here, with default values as well.
	def set_variables(self, S = NULL, kp = NULL, ki = NULL, kd = NULL, mu = NULL, g = NULL):
		## PID controller variables
		if self.is_valid(kp): self.kp = kp
		if self.is_valid(ki): self.ki = ki
		if self.is_valid(kd): self.kd = kd
		## input signal
		if self.is_valid(S ): self.S = S
		## pendulum variables
		if self.is_valid(g ): self.g = g 
		if self.is_valid(mu): self.mu = mu 	# dampenin

	def set_approximation(self, approximation = "euler"):
		if approximation in self.Approximations:
			self.approx = approximation
		else:
			print("Unknown numerical approximation technique")
		
	def set_integral(self, integral = "trapezoid"):
		if integral in self.Integrals:
			self.integral = integral
		else:
			print("Unknown integration approximation technique")

	## Function provided to reset the arrays to initial conditions based on what the current set values are.
	def zero_arrays(self):
		self.U = [self.kp] 	# Control signal
		self.Y = [0] 		# Position
		self.V = [0] 		# Velocity
		self.E = [self.S - self.Y[0]] 	# Error signal

	## Def internal functions for integration. Integration is for all of E.
	def nonlinear_update(self, timestep):
		# solve the ODE for angular position 
		t = [0, timestep]
		yv_sol = odeint (nonlinear_damped_pen_position,
		                [self.Y[-1], self.V[-1] ], 
		                t, 
		                args=(self.g, self.mu, self.U[-1]) )
		## Next position and velocity
		y_next, v_next = yv_sol[-1]
		self.Y.append(y_next)
		self.V.append(v_next)
		# Error signal
		e_next = self.S - y_next
		self.E.append(e_next)
		# Control signal
		if self.approx == "euler":
			if self.integral == "rectangle":
				int_area = integrate_sum_rectangle(self.E, timestep)
			elif self.integral == "trapezoid":
				int_area = integrate_sum_trapezoid(self.E, timestep)
			elif self.integral == "simpson":
				if time > 2:
					int_area = integrate_sum_simpson(self.E, timestep)
				else: # Approximate with trapezoid if less than 3 function values
					int_area = integrate_sum_trapezoid(self.E, timestep)

			u_next = (self.kp*e_next + self.ki*int_area + 
			          self.kd*((self.E[-1] - self.E[-2])/timestep))
		elif self.approx == "backwards":
			if time > 2:
				u_next = (self.U[-1] + self.kp*(self.E[-1] - self.E[-2]) + self.ki*self.E[-1]*timestep +
			    	      self.kd*(self.E[-1] - 2*self.E[-2] + self.E[-3])/timestep)
			else: # Approximate with trapezoid if less than 3 function values
				u_next = (self.kp*e_next + self.ki*integrate_sum_trapezoid(self.E, timestep) + 
			          self.kd*((self.E[-2] - self.E[-2])/timestep))
		self.U.append(u_next)

	def run(self, maxtime = 100, timestep = 0.1):
		for time in xrange(1, int(math.ceil(maxtime/timestep))):
			self.nonlinear_update(timestep)

	def results(self):  
		pid_dict = {"U":self.U,"Y":self.Y,"V":self.V,"E":self.E}
		return pid_dict


'''
Generate training or testing batch
'''
class batch( pendulum, PID_delay ):

	def __init__ (self, num_of_samples, maxtime, timestep):
		pendulum.__init__(self)
		PID_delay.__init__(self)
		self.num_of_samples = num_of_samples
		self.maxtime = maxtime
		self.timestep = timestep

	def pen_batch(self):
		self.batch_xs =  [ ] 
		self.batch_ys =  [ ]
		self.batch_ysl = [ ]
		pen = pendulum()
		for _ in range(self.num_of_samples):
			pen.zero_arrays()
			pen.run(self.maxtime, self.timestep)
			data = pen.results()
			self.batch_xs.append(copy(data["F"]))
			self.batch_ys.append(copy(data["Y_non"]))
			self.batch_ysl.append(copy(data["Y_lin"]))
		self.batch_xs = np.array(self.batch_xs)
		self.batch_ys = np.array(self.batch_ys)
		self.batch_ysl = np.array(self.batch_ysl)
		batch_dict = {"batch_xs": self.batch_xs, "batch_ys": self.batch_ys, "batch_ysl": self.batch_ysl}

		return batch_dict

	def pid_batch(self):
		self.batch_us = [ ]
		self.batch_es = [ ]
		self.batch_ys = [ ]
		self.batch_vs = [ ]
		pid = PID_delay ()
		for _ in range(self.num_of_samples):
			pid.zero_arrays()
			pid.run(self.maxtime, self.timestep)
			data = pid.results()
			self.batch_us.append(copy(data["U"]))
			self.batch_es.append(copy(data["E"]))
			self.batch_ys.append(copy(data["Y"]))
			self.batch_vs.append(copy(data["V"]))
		self.batch_us = np.array(self.batch_us)
		self.batch_es = np.array(self.batch_es)
		self.batch_ys = np.array(self.batch_ys)
		self.batch_vs = np.array(self.batch_vs)
		batch_dict = {"batch_us": self.batch_us, "batch_ys": self.batch_ys, "batch_es": self.batch_es, "batch_vs": self.batch_vs}

		return batch_dict

# batch_size = 50
# maxtime = 2
# timestep = 0.1
# num_steps = int(maxtime/timestep)
# xtrainset = np.empty([0, num_steps])
# ytrainset = np.empty([0, num_steps])
# batch = batch(batch_size, maxtime, timestep)
# epochs = 1000 

# for ind in range(epochs):
# 	print(ind)
# 	data = batch.pen_batch()
# 	xtrainset = np.append(xtrainset, data['batch_xs'], axis= 0)b
# 	ytrainset = np.append(ytrainset, data['batch_ys'], axis = 0)

# pickle.dump(xtrainset, open('xtrainset.p','wb'))
# pickle.dump(ytrainset, open('ytrainset.p','wb'))

# batch_size = 100
# maxtime = 2
# timestep = 0.1
# num_steps =int(maxtime/timestep)
# batch = batch(batch_size, maxtime, timestep)
# data = batch.pen_batch()
# xtestset = data['batch_xs']
# ytestset = data['batch_ys']
# yltestset = data['batch_ysl']
# pickle.dump (xtestset, open('xtestset.p','wb'))
# pickle.dump (ytestset, open('ytestset.p','wb'))
# pickle.dump (yltestset, open('yltestset.p','wb'))
