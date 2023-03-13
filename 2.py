import numpy as np
from simplex import Simplex

n, u, v = np.array([int(x) for x in input().split()])

m = u + v
inequality = [1] * u + [-1] * v

c = np.array([float(x) for x in input().split()])
A = np.array([[inequality[i] * float(x) for x in input().split()]
             for i in range(m)])

b = np.array([float(x) for x in input().split()])
b = np.append(b[:u], -b[u:])

###################################################

eta = 10**-10

while True:
    # print(A, b, c)

    optim = Simplex(A, b, c)
    optim.solve()
    # optim.output()

    # print(optim.table)

    if optim.status != optim.status.OPTIMAL:
        break  # no finite optimal solution
    
    eqn = optim.table[1:, -1]

    if np.allclose(eqn, np.floor(eqn), eta, eta):
        break  # integer solution found

    eqn = np.argmax(~np.isclose(eqn, np.floor(eqn), eta, eta))
    eqn = np.floor(optim.table[1 + eqn])

    coeff = -eqn[-optim.m-1:-1]
    constraint_A = np.sum(coeff[:, None] * A, axis=0) + eqn[:-optim.m-1]
    constraint_b = np.sum(coeff * b) + eqn[-1]

    A = np.append(A, [constraint_A], axis=0)
    b = np.append(b, [constraint_b])

    # print('----')

if optim.status == optim.Status.OPTIMAL:    
    res, x_optim = optim.get_solution()
    print(res)
    print(*x_optim[:optim.n-optim.m].astype(int))
elif optim.status == optim.Status.UNBOUNDED:
    print("Unbounded")
elif optim.status == optim.Status.INFEASIBLE:
    print("Infeasible")
