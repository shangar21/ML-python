import numpy as np 


def h(t1: float, t2: float, x: float):
	return t1 + (t2 * x)

def j(t1: float, t2:float, x: np.array, y: np.array, a: int, i: int):
	m = len(x)
	t1 = t1 - a*(1/m)*(h(t1, t2, x[i]) - y[i])
	t2 = t2 - a*(1/m)*(h(t1, t2, x[i]) - y[i])

def check_conv(t1, t2):
	return abs(t1) - abs(t2) < 0.001


t1 = 0
t2 = 1
i = 0
x = np.arange(20)
y = np.arange(20)
a = 0.5
conv = True

print(x)
print(y)

while(conv):
	j(t1, t2, x, y, a, i)
	conv = check_conv(t1, t2)
	i += 1

print("done!")

