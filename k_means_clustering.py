import numpy as np
import matplotlib.pyplot as plt

def get_mean():
	pass

def move(x, y, kp):
	pass

def is_conv():
	pass


sz=35

x = np.random.rand(sz)
y = np.random.rand(sz)

k=4
x_k = np.random.rand(k)
y_k = np.random.rand(k)

cp = {}
kp = {0:[], 1:[], 2:[]}



for i in range(sz):
	p_dist = []
	for j in range(k):
		p_dist.append(np.linalg.norm(np.array(x[i], y[i]) - np.array(x_k[j], y_k[j])))
	cp[i] = p_dist.index(min(p_dist))
	kp[p_dist.index(min(p_dist))] = kp.get(p_dist.index(min(p_dist)), [])
	kp[p_dist.index(min(p_dist))].append(i)

