import numpy as np
import matplotlib as plt

def normalize(arr: np.array()) -> np.array():
	normal = arr.T
	for i in normal:
		avg = sum(normal[i])
		max_min  = max(normal[i]) - min(normal[i])
		for j in normal[i]:
			j = (j - avg)/max_min

	return normal


