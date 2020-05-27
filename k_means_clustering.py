import numpy as np
import matplotlib.pyplot as plt

sz=35

x = np.random.rand(sz)
y = np.random.rand(sz)

k=4
x_k = np.random.rand(k)
y_k = np.random.rand(k)

for i in range(sz):
	for j in range(k):
		dist = numpy.linalg.norm(np.array(x[i], y[i]) - np.array(x_k[j], y_k[j]))
		print(dist)