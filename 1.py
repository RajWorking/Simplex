import numpy as np
from simplex import Simplex

n, u, v = np.array([int(x) for x in input().split()])

m = u + v
inequality = [1] * u + [-1] * v

c = np.array([float(x) for x in input().split()])
A = np.array([[inequality[i] * float(x) for x in input().split()] for i in range(m)])

b = np.array([float(x) for x in input().split()])
b = np.append(b[:u], -b[u:])

optim = Simplex(A, b, c)
optim.benchmark()
optim.solve()



