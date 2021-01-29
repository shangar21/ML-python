import numpy as np
from sklearn import datasets
import random
from scipy.special import expit
import csv
from sklearn import preprocessing
import matplotlib.pyplot as plt


def g(v):
	return expit(v)

def g_prime(v):
	tmp = np.ones(np.shape(v))
	return np.multiply(v, np.subtract(tmp,v))

def h(v):
	return np.maximum(np.zeros(v.shape),v)

def h_prime(v):
	tmp = []
	for i in v:
		if i > 0:
			tmp.append(1)
		else:
			tmp.append(0)
	return tmp

def k(v):
	return np.tanh(v)

def k_prime(v):
	return tmp - np.square(v)

def normalize(train_set):
	cols = np.transpose(train_set)
	mean = [sum(k)/len(k) for k in cols]
	sd = [max(k) - min(k) for k in cols]
	normalized = []

	for i in range(len(cols)):
		normalized.append([(k - mean[i])/sd[i] for k in cols[i]])

	return np.transpose(normalized)


class NN:
	
	weights = []
	hidden_layer_1 = []
	output_layer = []
	DELTA_2 = 0
	DELTA_1 = 0
	s1 = 0
	s2 = 0
	s3 = 0
	l = 0

	def __init__(self, s1, s2, s3, l):
		self.weights.append(np.random.rand(s2, s1))
		self.weights.append(np.random.rand(s3,s2+1))
		self.s1 = s1
		self.s2 = s2
		self.s3 = s3
		self.l = l

	def forward(self, input_layer):
		self.hidden_layer_1 = np.insert(input_layer, 0, 1)
		self.hidden_layer_1 = self.weights[0].dot(input_layer)
		self.hidden_layer_1 = h(self.hidden_layer_1)
		self.hidden_layer_1 = np.insert(self.hidden_layer_1, 0, 1)
		self.output_layer = self.weights[1].dot(self.hidden_layer_1)
		self.output_layer = h(self.output_layer)

	def backward(self, input_layer, y):
		del_out = self.output_layer - y
		del_2 = np.multiply(self.weights[1].T.dot(del_out), h_prime(self.hidden_layer_1))
		self.DELTA_2 += np.outer(del_out, self.hidden_layer_1)
		self.DELTA_1 += np.outer(del_2.reshape(self.s2+1, 1), input_layer.reshape(1,self.s1))

	def cost(self, train_set, y, m):
		p = []

		for i in range(len(train_set)):
			self.forward(train_set[i])
			p.append(np.square(y[i] - self.output_layer))

		return (1/m)*(np.sum(p))


	def adjust_weights(self, m):
		self.weights[0] = np.insert(self.weights[0], 0, np.zeros(self.s1))
		self.weights[0] = self.weights[0].reshape(self.s2+1, self.s1)

		D_out = (1/m)*self.DELTA_2  + (self.l)*(self.weights[1])
		D_in = (1/m)*self.DELTA_1 + (self.l)*(self.weights[0])

		self.weights[0] = np.subtract(self.weights[0], D_in)
		self.weights[1] = np.subtract(self.weights[1], D_out)

		self.weights[0] = self.weights[0][1:]

		self.DELTA_2 = 0
		self.DELTA_1 = 0


results = []

with open("DataSets/diabetes.csv") as csvfile:
	reader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)
	for row in reader:
		results.append(np.array(row))

results = np.array(results)
y = results[:, [-1]]
results = np.delete(results, -1, 1)
results = normalize(results)

net = NN(8,12,1,1.3)
cost = []


for i in range(100):
	for i in range(0, len(results), 3):
		net.forward(results[i])
		net.backward(results[i], y[i])
	net.adjust_weights(len(results)//3)
	cost.append(net.cost(results[:len(results)//3], y[:len(results)//3], len(results)//3))
	print(net.cost(results[:len(results)//3], y[:len(results)//3], len(results)//3))
plt.plot(cost)
plt.show()




		
# p = []
# for i in range(len(train_set)):
# 	self.forward(train_set[i])
# 	p.append(np.multiply(-y[i], np.log(self.output_layer) 
# 		+ np.multiply(1-y[i], np.log(1-self.output_layer))))
# p = np.sum(p)
# square_weights = []
# for i in self.weights:
# 	square_weights.append(np.sum(np.square(i)))

# square_weights = sum(square_weights)
# square_weights *= self.l/(2*m)

# p += square_weights		
# return p
# instead of the approach from andrew ng's course I will try a MSE instead


# for i in range(3):
# 	for i in range(3):
# 		net.forward(results[i])
# 		net.backward(results[i], y[i])
# 	net.adjust_weights(3)


#self.DELTA_2 += self.hidden_layer_1.T.dot(del_out[0])
#self.DELTA_1 += del_2.reshape(self.s2+1,1).dot(input_layer.reshape(1,self.s1))