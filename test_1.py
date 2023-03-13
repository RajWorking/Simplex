import numpy as np
from simplex import Simplex

import random


def test():
    n = random.randrange(1, 50)
    m = random.randrange(1, 30)
    A = np.random.rand(m, n)
    b = np.random.rand(m) * 1000 * random.randrange(-1,2,2)
    c = -np.random.rand(n)

    print(n, m)
    # print("A:", A)
    # print("b:", b)
    # print("c:", c)

    optim = Simplex(A, b, c)
    optim.benchmark()
    optim.solve()
    optim.output()


while True:
    test()

    print("Press 1 to continue, 0 to stop.")
    ok = int(input())
    if ok != 1:
        break
