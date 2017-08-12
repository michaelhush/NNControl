'''
Implement  KNN class

Inputs: 
k = number of neighbours
norm = string of choosing distance metrics 
xtrainset, ytrainset = batch of training samples 
xtest, ytest = batch of testing samples
num_steps = length of input time series
compare = string of "both" or "position"

Outputs: 
KNN_cost = cost of KNN predictions

To generate xtrainset, ytrainset, must define
xtrainset = np.empty([0,num_steps]) 
ytrainset = np.empty([0,num_steps])
Then np.append(xtrainset, batch_xs, axis=0)
'''
import math as math
import numpy as np

class KNN (object):
	Norm = ["Euclidean", "Chebychev", "Manhattan"]
	def __init__(self, k, norm, xtrainset, ytrainset, num_steps):
		self.k = k  ## number of neighbours
		if norm in self.Norm:
			self.norm = norm 
		self.xtrainset = xtrainset
		self.ytrainset = ytrainset
		self.num_steps = num_steps

	def distance(self, train, test):
		if self.norm == "Euclidean":
			dist = [(train[i] - test[i])*(train[i] - test[i]) for i in range(len(train))]
			dist = math.sqrt(sum(dist))

		elif self.norm == "Manhattan":
			dist = [abs(train[i] - test[i]) for i in range(len(train))]
			dist = sum(dist)

		else: 
			dist = [abs(train[i] - test[i]) for i in range(len(train))]
			dist = max(dist)
		return dist

	def kneighbours(self, test):
		distances = [(self.ytrainset[i].tolist(), self.distance(self.xtrainset[i], test) ) for i in range(len(self.xtrainset))]
		dtype = [('ytrainset', list), ('dist', float) ]
		distances = np.array(distances, dtype = dtype)
		distances = np.sort(distances, order = 'dist')
		neighbours = [ distances[i][0] for i in range(self.k)]
		return neighbours

	def predictKNN(self, neighbours):
		weights = [float(2**(-i)) for i in range(self.k)]
		posweight = np.empty([self.num_steps, self.k])
		velweight = np.empty([self.num_steps, self.k])
		for i in range(self.num_steps):
			posweight[i] = [p[i]*w for p,w in zip(neighbours, weights)]
		predictpos = [ sum(posweight[i])/sum(weights)  for i in range(self.num_steps)]
		return predictpos

	def KNN_cost(self, xtest, ytest):
		test_KNN_cost = [ ]
		for j in range(len(xtest)):
			neig = self.kneighbours(xtest[j])
			ykt_ = self.predictKNN(neig)
			test_KNN_cost.append([ (ykt_[i] - ytest[j][i])*(ykt_[i] - ytest[j][i]) for i in range(len(ykt_)) ])

		test_KNN_cost = np.array(test_KNN_cost)
		KNN_cost = np.mean(test_KNN_cost)

		return KNN_cost

	# def main():
		
