## class for PID simulation
## PID with a damped pendulum
from __future__ import division
import matplotlib.pyplot as plt
from math import ceil
import numpy as np
from scipy.integrate import odeint

NULL = "" # THIS IS KIND OF DUMB

## Damped pendulum position

## Damped Pendulum equations of motion
# y1 = angular position
# y2 = angular velocity
def nonlinear_damped_pen_position (y,t,g,mu,u):
    y1,y2 = y
    dydt = [y2, -g*np.sin(y1) - mu*y2  + u ]
    return dydt

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
class pid(object):
	Approximations = ["euler", "backwards"]
	Integrals      = ["rectangle", "trapezoid", "simpson"]
	## Initialize a class with given maxtime and timestep
	def __init__(self):
		# self.set_time()
		self.init_variables()
		self.zero_arrays()
		self.set_approximation()
		self.set_integral()	

	def is_valid(self, a):
		return isinstance(a, int) or isinstance(a, float)

	def init_variables(self):
		self.set_variables(S = 0, kp = 1, ki = 6, kd = 4, mu = 0.1, g= 9.8)

	## Set variables here, with default values as well.
	def set_variables(self, S = NULL, kp = NULL, ki = NULL, kd = NULL, mu = NULL, g = NULL):
		## PID controller variables
		if self.is_valid(kp): self.kp = kp
		if self.is_valid(ki): self.ki = ki
		if self.is_valid(kd): self.kd = kd
		## input signal
		if self.is_valid(S ): self.S = S
		## pendulum variables
		# if self.is_valid(k ): self.k  = k 	# spring constant
		# if self.is_valid(m ): self.m  = m 	# mass of pendulum
		if self.is_valid(g ): self.g = g 
		if self.is_valid(mu): self.mu = mu 	# dampening
		# if self.is_valid(f ): self.f  = f 	# frictional force
		# if self.is_valid(c ): self.c  = c 	# non-linear term in angular momentum

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
		# self.Z = [0] 		# angular position
		# self.O = [0] 		# angular velocity
		self.E = [self.S - self.Y[0]] 	# Error signal

	## Def internal functions for integration. Integration is for all of E.
	def nonlinear_update(self, timestep):
		## solve the ODE for angular position 
		t = [0, timestep]
		yv_sol = odeint (nonlinear_damped_pen_position,
		                [self.Y[-1], self.V[-1] ], 
		                t, 
		                args=(self.g, self.mu, self.U[-1]) )
		## Next position and velocity
		y_next, v_next = yv_sol[-1]
		self.Y.append(y_next)
		self.V.append(v_next)
		## Error signal
		e_next = self.S - y_next
		self.E.append(e_next)
		## Control signal
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
		for time in xrange(1, int(ceil(maxtime/timestep))):
			self.nonlinear_update(timestep)

	def results(self):  
		pid_dict = {"U":self.U,"Y":self.Y,"V":self.V,"E":self.E}
		return pid_dict

p = pid()
p.set_variables(S=0.1, kp = 0.06, kd= 0.24 , ki = 0.06) 
p.run(1,1)

d = p.results()
# Time vector may not be linearly spaced
maxtime = 1
timestep = 1
T = np.linspace(0, maxtime, int(ceil(maxtime/timestep)))
print (len(d["E"]))

# p = pid()
# p.set_variables(S=0.1, kp = 1, kd= 4 , ki = 6) 
# maxtime = 5
# timestep = 0.1
# p.run(maxtime,timestep)
# d= p.results()
# T = np.linspace(0,maxtime, int(ceil(maxtime/timestep)))
# plt.figure(1)
# plt.plot(T,d["Y"])
# plt.show()

# print (d["E"])
# print "---------"
# print (d["Y"])
