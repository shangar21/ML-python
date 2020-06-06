import numpy as np

# Simple 2 Layer Neural network


class NN:
	def __init__(self, data_in, expected):
		self.data_in = data_in
		self.weights1 = np.random.rand(self.input.shape[1], 4) #initializing random set of weights for hidden layer
		self,weights2 = np.random.rand(4,1) #weights that take hidden layer input to the output layer, from 4 to 1
		self.expected = expected
		self.output = np.zeros(y.shape) 

	def forward_prop(self):
		self.layer1 = sigmoid(np.dot(self.input, self.weights1))
		self.output = sigmoid(np.dot(self.layer1, self.weights2))

	def back_prop(self):
		d_out = self.output - self.expected
		d_weights2 = (weights2.dot(d_out)).multiply(sigmoid_derivative(self.output))
		d_weights1 = (weights1.dot(d_weights2)).multiply(sigmoid_derivative(self.layer1))
		self.weights1 += d_weights1
		self.weights2 += d_weights2


