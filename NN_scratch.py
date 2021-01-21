import numpy as np
from sklearn import datasets
import random

# Extremely rudimentary example of a neural network. This was simply to understand how tf back prop works cuz I had bare trouble with it. Will try again but with an actual dataset so lets see how smart ya boi is. 


# THETA = np.random.randn(3,4)
# THETA_2 = np.random.randn(1,4)
# x = np.array([1,2,3,4])

# activation_layer_1 = []

# # #NONVECTORIZED IMPLETENTAION
# # for i in range(len(x)-1):
# # 	activation_layer_1.append(g(x.dot(THETA[i])))
# # activation_layer_1 = np.insert(activation_layer_1, 0, 1)
# # output_layer = g(activation_layer_1.dot(THETA[0]))
# # print(output_layer)

# #VECTORIZED IMPLEMENTAION
# activation_layer_1 = THETA.dot(x)
# p_out = THETA_2.dot(activation_layer_1)
# out = g(p_out[0][0]) 

# cancer = datasets.load_breast_cancer()
# X = cancer.data
# Y = cancer.target
# c = list(zip(X,Y))
# random.shuffle(c)
# X,Y = zip(*c)

# # Simple 2 Layer Neural network

def g(v):
	return (1/(1 + np.exp(-v)))

def g_prime(v):
	tmp = np.ones(np.shape(v))
	return np.multiply(v, np.subtract(tmp,v))

class NN:
	weights = []
	input_layer_size = 0
	output_layer_size = 0
	activation_layer_1 = []
	output_layer= []

	def __init__(self,s1,s2,s3):
		self.input_layer_size = s1
		self.output_layer_size = s3
		self.weights.append(np.random.randn(s2,s1))
		self.weights.append(np.random.randn(s3,s2 + 1))

	def forward(self, input_layer):
		self.activation_layer_1 = self.weights[0].dot(input_layer)
		self.activation_layer_1 = np.insert(self.activation_layer_1,0,1)
		self.output_layer = self.weights[1].dot(self.activation_layer_1)
		self.output_layer = g(self.output_layer)

	def cost(self):
		# J(T) = -1/m(sum_i^m(sum_j^K(y_i[j]log(h_t(x_i)[j])+(1-y_i[j])log(1-(h_t(x_i)[j]))))) + l/2m(sum_l^{L-1}(sum_i^{s_l}(sum_j^{sl+1}(T)_l[j][i])))
		# In this neural net there are only 3 layers, thus L = 3. The output layer will only have one element thus K = 1. s_1 = 4, s_2 = 3. This is a dummy example with only one training example. m = 1. Will try a different one with the breast cancer one if this works. 
		# because our training set's expected output of y is 0, and the number of training sets is 1 we can simplify above to -log(1-h_t(x))+ l/2m(sum_l^{L-1}(sum_i^{s_l}(sum_j^{sl+1}(T)_l[j][i])))where x is the output layer, which in this case is only 1 number in [0,1].  
		cost = np.log(1-self.output_layer[0])
		s = [np.square(k) for k in self.weights]
		s = [sum(k) for k in self.weights]
		s = [sum(k) for k in s]
		s = sum(s)
		return cost + s 

	def back(self, y, x):
		delta_out = self.output_layer - y
		delta_2 = np.multiply(self.weights[1].T.dot(delta_out), g_prime(self.activation_layer_1))
		DELTA_2 = self.activation_layer_1.T.dot(delta_out[0])
		DELTA_1 = x.dot(delta_2)
		self.weights[0] = np.subtract(self.weights[0], DELTA_1)
		self.weights[1] = np.subtract(self.weights[1], DELTA_2)


x = np.array([1,2,3,4])
y = 0
net = NN(4,3,1)
print(net.weights)
net.forward(x)
net.back(y,x)
print(net.weights)
net.forward(x)
#net.back(y,x)
print(net.output_layer)
#The shit actually prints 0 on god. Im shocked that my thing ac worked :>)