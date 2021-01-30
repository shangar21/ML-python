import numpy as np
from sklearn import datasets
import random
from scipy.special import expit
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelBinarizer
from scipy.special import expit, logit

#sigmoid
def g(v):
	return expit(v)

def g_prime(v):
	return v * (1 - v)

class NN:

	s1 = 0
	s2 = 0
	s3 = 0
	lr = 0
	DELTA_HL_OUT = 0
	DELTA_IN_HL = 0
	weights = []
	hidden_layer_1 = []
	output_layer = []

	def __init__(self, s1, s2, s3, lr):
		self.s1 = s1
		self.s2 = s2
		self.s3 = s3
		self.lr = lr
		self.weights.append(np.random.rand(s1, s2))
		self.weights.append(np.random.rand(s2, s3))

	def forward(self, input_layer):
		self.hidden_layer_1 = input_layer.dot(self.weights[0])
		self.hidden_layer_1 = g(self.hidden_layer_1)
		self.output_layer = self.hidden_layer_1.dot(self.weights[1])
		self.output_layer = g(self.output_layer)

	def backward(self, x_train, y_train):
		output_error = y_train - self.output_layer
		del_out = output_error*g_prime(self.output_layer)
		del_hl = output_error.dot(self.weights[1].T)*g_prime(self.hidden_layer_1)

		self.weights[0] += self.lr*x_train.reshape(5,1).dot(del_hl.reshape(1,4))
		self.weights[1] += self.lr*self.hidden_layer_1.reshape(4,1).dot(del_out.reshape(1, 3))


	def adjust_weights(self, m):
 		pass


 	# def cost(self, x_train, y_train):
 	# 	MSE = []
 	# 	for i in range(len(x_train)):
 	# 		self.forward(x_train)
 	# 		MSE.append(np.square(self.output_layer - y_train[i]))

 	# 	return (1/len(x_train))*np.sum(MSE)		


data = datasets.load_iris()

scaler = StandardScaler()
lb = LabelBinarizer()

x_train, x_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.33, random_state=42)

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

y_train = lb.fit_transform(y_train)
y_test = lb.transform(y_test)

net = NN(5,4,3,3)

for i in range(1000):
	for i in range(len(x_train)):
		net.forward(np.insert(x_train[i], 0, 1))
		net.backward(np.insert(x_train[i], 0, 1), y_train[i])

count = 0

for i in range(len(x_test)):
	net.forward(np.insert(x_test[i], 0, 1))
	hypothesis = net.output_layer
	for j in range(len(hypothesis)):
		if hypothesis[j] <= 0.01:
			hypothesis[j] = 0
		elif hypothesis[j] >= 0.9:
			hypothesis[j] = 1
	if np.array_equal(np.array(hypothesis), y_test[i]):
		count+= 1	

print(count/len(x_test))