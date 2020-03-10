import numpy as np
import math

def g(x, t):
	z = x.dot(t)
	for i in z:
		z[i] = 1/1 + math.exp(-z[i])
	return z



x = np.genfromtxt('ufwbcd.csv', comments=None, delimiter=',')

# results, benign = 2, malignant = 4
y = x[:, 0]

x = np.delete(x,0,axis=1)
x = np.c_[np.ones(x.shape[0]), x]

a = 0.5
m = x.shape[1]
t = np.zeros(x.shape[1])
dv = np.array()
conv = False

while not conv:
	dv = (x.T.dot((g(x,t)) - y)
	t = t - (a/m)*(dv)
	conv = np.vectorize(abs)(dv).sum() <0.0001


print("conv!")






