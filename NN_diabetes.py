import numpy as np
from sklearn import datasets
import random
from scipy.special import expit
import csv
from sklearn import preprocessing

def g(v):
	return expit(v)

def g_prime(v):
	tmp = np.ones(np.shape(v))
	return np.multiply(v, np.subtract(tmp,v))


class NN:
	
	weights = []
	hidden_layer_1 = []
	output_layer = []
	DELTA_2 = 0
	DELTA_1 = 0
	s1 = 0
	s2 = 0
	s3 = 0

	def __init__(self, s1, s2, s3):
		self.weights.append(np.random.randn(s2, s1))
		self.weights.append(np.random.randn(s3,s2+1))
		self.s1 = s1
		self.s2 = s2
		self.s3 = s3

	def forward(self, input_layer):
		self.hidden_layer_1 = self.weights[0].dot(input_layer)
		self.hidden_layer_1 = g(self.hidden_layer_1)
		self.hidden_layer_1 = np.insert(self.hidden_layer_1, 0, 1)
		self.output_layer = self.weights[1].dot(self.hidden_layer_1)
		self.output_layer = g(self.output_layer)

	def backward(self, input_layer, y):
		del_out = self.output_layer - y
		del_2 = np.multiply(self.weights[1].T.dot(del_out), g_prime(self.hidden_layer_1))
		self.DELTA_2 += self.hidden_layer_1.T.dot(del_out[0])
		self.DELTA_1 += del_2.reshape(self.s2+1,1).dot(input_layer.reshape(1,self.s1))

	def cost(self, train_set, y, m):
		
		p = []
		for i in range(len(train_set)):
			self.forward(train_set[i])
			p.append(np.multiply(y[i], np.log(self.output_layer) + np.multiply(1-y[i], np.log(1-self.output_layer))))

		return (-1/m)*sum(p)


	def adjust_weights(self, m):
		self.weights[0] = np.insert(self.weights[0], 0, np.zeros(self.s1))
		self.weights[0] = self.weights[0].reshape(self.s2+1, self.s1)
		self.weights[0] = np.subtract(self.weights[0], (1/m)*self.DELTA_1)
		self.weights[1] = np.subtract(self.weights[1], (1/m)*self.DELTA_2)
		self.weights[0] = self.weights[0][1:]


results = []

with open("DataSets/diabetes.csv") as csvfile:
	reader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)
	for row in reader:
		results.append(np.array(row))

results = np.array(results)

scaler = preprocessing.StandardScaler().fit(results)
results = scaler.transform(results)


y = results[:, -1]
results = np.delete(results, -1, 1)

c = list(zip(results, y))
random.shuffle(c)
results, y = zip(*c)

net = NN(8,5,1)

for i in range(30):
	for i in range(len(results)):
		net.forward(results[i])
		net.backward(results[i], y[i])
	net.adjust_weights(len(results))
	print(net.cost(results, y, len(results)))

count = 0
total = len(results)

for i in range(len(results)):
	net.forward(results[i])
	if net.output_layer[0] - y[i] < 0.001:
		count += 1

print(count/total)


