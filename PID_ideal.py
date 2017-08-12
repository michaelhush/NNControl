# class for PID simulation
from __future__ import division
import matplotlib.pyplot as plt
from math import ceil
import numpy as np
from scipy.integrate import odeint

NULL = "" # THIS IS KIND OF DUMB

# Damped pendulum position
## This is a simpler model with no delay
## See notes for notation and detail
# g1 = g, g2 = g' =g1, g3 =g''=g2
def damped_pen_position(g, t, zeta, omega, kd, kp, ki, S):
	g1, g2, g3 = g
	dgdt = [g2,g3, -(2*zeta*omega + kd)*g3 - (omega**2 +kp)*g2 - ki*g1 + S]
	return dgdt

# PID class

class PID_ideal(object):

	# initialize a class with given maxtime and timestep
	def __init__(self):
		# self.set_time()
		self.ran = False
		self.init_variables()
	# def set_time(self, maxtime = 100, timestep = 0.1):
	# 	self.maxtime = maxtime
	# 	self.timestep = timestep
	def is_valid(self, a):
		return isinstance(a, int) or isinstance(a, float)

	def init_variables(self):
		self.set_variables(S = 1, kp = 1.0, ki = 8.0, kd = 15.0, zeta =1/(2*np.sqrt(20)) , omega= np.sqrt(20))

	# set variables here, with default values as well.
	def set_variables(self, S = NULL, kp = NULL, ki = NULL, kd = NULL, zeta = NULL, omega = NULL):
		# PID controller variables

		if self.is_valid(kp): self.kp = kp
		if self.is_valid(ki): self.ki = ki
		if self.is_valid(kd): self.kd = kd
		# input signal
		if self.is_valid(S ): self.S = S
		# pendulum variables
		if self.is_valid(zeta): self.zeta  = zeta	# damping rate
		if self.is_valid(omega): self.omega = omega 

	# function provided to reset the arrays to initial conditions based on what the current set values are.
	def zero_arrays(self):
		self.Y = [0] # Position
		self.G1 = [0]
		self.G2 = [0]
		self.G3 = [0]

	# def internal functions for integration. Integration is for all of E.
	def update(self, timestep):
		# solve the single ODE for position 
		t = [0, timestep]
		g_sol = odeint(damped_pen_position, 
						[self.G1[-1], self.G2[-1], self.G3[-1]], t, 
						args = (self.zeta, self.omega, self.S, self.kd, self.ki, self.kp))
		g1_next, g2_next, g3_next = g_sol[-1]
		self.G1.append(g1_next)
		self.G2.append(g2_next)
		self.G3.append(g3_next)

		# Calculate the output signal
		y_next = self.kd*g3_next + self.kp*g2_next + self.ki*g1_next
		self.Y.append(y_next)

	def reset(self):
		self.ran = False

	def run(self, maxtime = 100, timestep = 0.1):
		if not self.ran:
			self.zero_arrays()
			self.ran = True
			start = 1
		else:
			start = 0

		for time in xrange(start, int(ceil(maxtime/timestep))):
				self.update(timestep)

	def results(self):  
		pid_dict = {"Y":self.Y, "G1":self.G1, "G2":self.G2, "G3":self.G3}
		return pid_dict


p = PID()
maxtime = 200
timestep = 0.1
T = np.linspace(0, maxtime, int(ceil(maxtime/timestep)))
# p.set_variables(S=0)

# Generate a random array 200 by 1 
# random = np.random.rand(200,1)

# for s in random[::]:
# 	p.set_variables(S=s)
# 	p.run(int(maxtime/timestep),1)


# p.run(20,0.1)
STEPS = 200 # make sure it's a factor of maxtime for now, otherwise T needs to be changed a bit.
values = np.linspace(-1, 1, STEPS)
for s in values[::-1]:
	p.set_variables(S = s)
	p.run(int(maxtime/STEPS), 0.1)
d = p.results()
plt.plot(T, d["Y"])
plt.show()
