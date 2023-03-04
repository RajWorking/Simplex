import numpy as np

'''
TODO: 
1. implement cycling
'''


class Simplex:

    def __init__(self, a, b, c):
        self.A = a
        self.b = b
        self.c = c

        self.n = self.c.size  # n: number of variables
        self.m = self.b.size  # m: number of equations

    def visualize(self):
        print(np.append(self.A, np.array([self.b]).T, axis=1))

    def standard(self):
        self.n += self.m
        self.c = np.append(self.c, np.zeros(self.m))
        self.A = np.append(self.A, np.eye(self.m), axis=1)

    def solve(self):
        self.standard()
        self.visualize()
        
    def two_phase(self):
        pass

    def benchmark(self):
        '''
        Checking actual solution using scipy
        Must be run before solve.
        '''
        import scipy
        res = scipy.optimize.linprog(self.c, A_ub=self.A, b_ub=self.b)
        print(res.fun)
        print(res.x)
