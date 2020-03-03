
import numpy as np
import matplotlib.pyplot as plt
import time 


#already added a row of ones at the bottom of the CSV
x = np.genfromtxt('mlr09.csv', comments=None, delimiter=',', skip_header=1)

y = x[:, 4]

W = np.ones(x.shape[1])

mu = 0.5

deriv = W.T.dot(x.T.dot(x)) - y.T.dot(x)

converged = false

while(not converged):
	deriv = W.T.dot(x.T.dot(x)) - y.T.dot(x)
	W = W - mu * deriv
	converged = np.vectorize(lambda x: abs(x))(derivative).sum() < 0.000001
		

print("Converged!")


