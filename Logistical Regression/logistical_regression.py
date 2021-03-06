import numpy as np
from sklearn import preprocessing 
import math

def g(x):
	return 1/(1 + np.exp(-x))


x = np.genfromtxt('ufwbcd.csv', delimiter=',')

# results, benign = 2, malignant = 4
y = x[:, 0]
y = np.vectorize(lambda x: 0 if x==2 else 1)(y)
x = np.delete(x,0,axis=1)
x = np.c_[np.ones(x.shape[0]), x]
a = 0.03
m = x.shape[1]
t = np.zeros(x.shape[1])
avg = np.mean(x, axis=0)
x -= avg
x /= 9
conv = False

while not conv:
	dv = x.T.dot(g(x.dot(t)) - y)
	t = t - (a/m)*dv
	conv = np.vectorize(abs)(dv).sum() <0.001
	print(dv)
	

clump_thickness = (int)(input("enter clump thickness: "))
uniformity_cell_size = (int)(input("enter uniformity of cell size: "))
cell_shape = (int)(input("enter uniformity of cell shape: "))
adhesion = (int)(input("enter marginal adhesion: "))
epithelial_cell_size = (int)(input("enter single epethelial cell size: "))
nuclei = (int)(input("enter bare nuclei: "))
chromatin = (int)(input("enter bland chromatin: "))
nucleoli = (int)(input("enter normal nucleoli: "))
mitosis = (int)(input("enter mitosis:  "))

test_results = [1, clump_thickness, uniformity_cell_size, cell_shape, adhesion, epithelial_cell_size, nuclei, chromatin, nucleoli, mitosis]
test_results -= avg
test_results /= 9

prediction = (g(test_results.dot(t)))

if prediction <= 0.4:
	print("benign")
elif prediction <= 0.8:
	print("maybe malignant")
else:
	print("malignant")







