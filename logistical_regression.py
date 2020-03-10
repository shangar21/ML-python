import numpy as np
import math

def g(x):
	for i in x:
		x[i] = 1/1 + math.exp(-x[i])
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
	dv = (x.T.dot((g(x.dot(t))) - y)
	t = t - (a/m)*(dv)
	conv = np.vectorize(abs)(dv).sum() <0.0001


clump_thickness = (int)(input("enter clump thickness: "))
uniformity_cell_size = (int)(input("enter uniformity of cell size: "))
cell_shape = (int)(input("enter uniformity of cell shape: "))
adhesion = (int)(input("enter marginal adhesion: "))
epithelial_cell_size = (int)(input("enter single epethelial cell size: "))
nuclei = (int)(input("enter bare nuclei: "))
chromatin = (int)(input("enter bland chromatin: "))
nucleoli = (int)(input("enter normal nucleoli: "))
mitosis = (int)(input("enter mitosis: "))

test_results = [1, clump_thickness, uniformity_cell_size, cell_shape, adhesion, epithelial_cell_size, nuclei, chromatin, nucleoli, mitosis]

prediction = g(test_results.dot(t))

print(prediction)







