import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import  distance

def move(kp, x_k, y_k):
	k_dict = {}
	for i in kp:
		x_sum = 0
		y_sum = 0
		numitems = 0
		for j in kp[i]:
			x_sum += j[0]
			y_sum += j[1]
			numitems += 1
		x_k[i] = x_sum/numitems
		y_k[i] = y_sum/numitems

sz=35

x = np.random.rand(sz)
y = np.random.rand(sz)

k=4
x_k = np.random.rand(k)
y_k = np.random.rand(k)

cp = {}
kp = {}

conv = False

while not conv:
	for i in range(sz):
		p_dist = []
		for j in range(k):
			p_dist.append(distance.euclidean(np.array([x[i], y[i]]), np.array([x_k[j], y_k[j]])))

		min_dist = p_dist.index(min(p_dist))
		cp[i] = min_dist
		kp[min_dist] = kp.get(min_dist, [])
		kp[min_dist].append([x[i], y[i]])

		pre_x, pre_y = x_k, y_k
		move(kp, x_k, y_k)

		conv = np.array_equiv(pre_x, x_k) and np.array_equiv(y_k, pre_y)

plt.scatter(x,y)
plt.scatter(x_k, y_k)
plt.show()
