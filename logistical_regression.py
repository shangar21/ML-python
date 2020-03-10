import numpy as np

def g(x, t):
	z = x.dot(t)
	return 1

x = np.genfromtxt('ufwbcd.csv', comments=None, delimiter=',')
# results, benign = 2, malignant = 4
y = x[:, 0]

x = np.delete(x,0,axis=1)
x = np.c_[np.ones(x.shape[0]), x]

a = 0.5
m = x.shape[1]
t = np.zeros(x.shape[1])
conv = False

while not conv:
	dv = x.T.dot()






