import numpy as np 


def h(t1: float, t2: float, x: float):
	return t1 + (t2 * x)

def j(t1: float, x: np.array, y: np.array, a: int, i: int):
	m = len(x)
	return t1 - a*(1/m)*(h(t1, t2, x[i]) - y[i])

def check_conv(t1, t2):
	return abs(t1) - abs(t2) < 0.001


t1 = 0
t2 = 1
i = 0
x = np.arange(-10,10)
y = [k *k for k in x]
a = .5
conv = check_conv(t1, t2)

print(x)
print(y)

while(conv and i < len(x)):
	t1 = j(t1, x, y, a, i)
	t2 = j(t2, x, y, a, i)
	conv = check_conv(t1, t2)
	print(t1, " ,", t2, " ,", abs(t1 - t2))
	i += 1

if (conv):
	print("found min")


