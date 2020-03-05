import numpy as np


x = np.genfromtxt('mlr09.csv', comments=None, delimiter=',', skip_header=1)
y = x[:, 4]
x = np.delete(x,4,axis=1)
x = np.c_[np.ones(x.shape[0]), x]
p_inv = np.linalg.inv(x.T.dot(x))
p_inv = p_inv.dot(x.T).dot(y)


height = float(input('enter height in feet (format: feet.inches): '))
weight = float(input('enter weight in lbs: '))
fg = float(input('enter field goal percentage: '))
ft = float(input('enter free throw percentage: '))
print('predicted average points per game: ', np.array([1, height, weight, fg, ft]).dot(p_inv.T))
