
import numpy as np
import matplotlib.pyplot as plt
import time 


x = np.genfromtxt('mlr09.csv', comments=None, delimiter=',', skip_header=1)

y = x[:, 4]

x = np.delete(x, 4, axis=1)

x = np.c_[x, np.ones(x.shape[0])]

W = np.zeros(x.shape[1])

mu = 0.5

converged =  False

<<<<<<< HEAD
while not converged:
	deriv = W.T.dot(x.T.dot(x)) - y.T.dot(x)
	W = W - ((mu) * (deriv))
	converged = np.vectorize(lambda x: abs(x))(deriv).sum() < 0.000001

=======
converged = false

while(not converged):
	deriv = W.T.dot(x.T.dot(x)) - y.T.dot(x)
	W = W - mu * deriv
	converged = np.vectorize(lambda x: abs(x))(derivative).sum() < 0.000001
		
>>>>>>> d495dfa380f0f8aa37e0b27ceb85527e195b86d5

print("Converged!")


