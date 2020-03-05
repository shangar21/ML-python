import numpy as np


x = np.genfromtxt('mlr09.csv', comments=None, delimiter=',', skip_header=1)
y = [:, 4]
x = np.delete(x,4,axis=1)
x = np.c_[np.ones(x.shape[0]), x]

p_inv = np.linalg.inv(x.T.dot(x))

p_inv = p_inv.dot(x.T).dot(y)
