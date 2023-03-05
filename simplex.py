import numpy as np
import enum

np.set_printoptions(linewidth=np.inf, suppress=True)


class Simplex:

    def __init__(self, a, b, c, debug=False):
        self.A = a
        self.b = b
        self.c = c
        self.debug = debug

        self.n = self.c.size  # n: number of variables
        self.m = self.b.size  # m: number of equations

        # list of basic variables
        self.basic_var = np.empty((0, self.m), dtype=np.int32)
        self.status = self.Status.SOLVING
        self.blands_rule = False

    class Status(enum.Enum):
        SOLVING = 0
        OPTIMAL = 1
        INFEASIBLE = 2
        UNBOUNDED = 3

    def visualize(self):
        print(self.table)

    def standardize(self):
        self.n += self.m
        self.c = np.append(self.c, np.zeros(self.m))
        self.A = np.append(self.A, np.eye(self.m), axis=1)

    def tabularize(self):
        ''' Convert to tableau format
        '''
        obj_row = np.append(-self.c, [0])
        self.table = np.append(self.A, np.array([self.b]).T, axis=1)
        self.table = np.append([obj_row], self.table, axis=0)

    def get_solution(self):
        ''' Solution from tableau
        '''
        x = np.zeros(self.n)
        x[self.basic_var[-1]] = self.table.T[-1][1:]
        return self.table[0][-1], x[:self.n-self.m]

    def output(self):
        if self.status == self.Status.OPTIMAL:
            res, x_optim = self.get_solution()
            print(res)
            print(*x_optim)
        elif self.status == self.Status.UNBOUNDED:
            print("Unbounded")

    def detect_cycle(self, arr, ele):
        if self.blands_rule:
            return
        if np.any([set(x) == set(ele) for x in arr]):
            self.blands_rule = True

    def iteration(self):
        basis = self.basic_var[-1]
        if self.debug:
            print('Basis', basis)

        self.detect_cycle(self.basic_var, basis)

        obj_row = self.table[0][:-1]
        entering_var = np.argmax(
            obj_row > 0) if self.blands_rule else np.argmax(obj_row)
        if self.table[0][entering_var] <= 0:
            # all negative coefficients in row
            self.status = self.Status.OPTIMAL
            return

        if self.debug:
            print('Entering Variable', entering_var)

        y = self.table.T[entering_var][1:]

        if np.all(y <= 0):
            # all negative coefficients in col
            self.status = self.Status.UNBOUNDED
            return

        x_b = self.table.T[-1][1:]

        ratios = np.array(
            [x_b[i]/y[i] if y[i] > 0 else np.inf for i in range(self.m)])
        leaving_var = np.argmin(ratios)

        if self.debug:
            print('Leaving Variable', basis[leaving_var])

        base_row = self.table[leaving_var + 1]
        base_row /= base_row[entering_var]
        self.table[leaving_var + 1] = base_row

        # Pivoting operations
        for i in range(self.m + 1):
            if i != leaving_var + 1:
                coeff = self.table[i][entering_var]
                self.table[i] -= coeff * base_row

        basis[leaving_var] = entering_var
        self.basic_var = np.append(self.basic_var, [basis], axis=0)
        return

    def two_phase(self):
        # TODO: introduce artifical variables
        pass

    def solve(self):
        self.standardize()
        self.tabularize()

        self.basic_var = np.append(
            self.basic_var, [np.arange(self.n - self.m, self.n)], axis=0)

        iter = 1
        while self.status == self.Status.SOLVING:
            if self.debug:
                print('Iteration:', iter)
                self.visualize()
                print('----')
            self.iteration()
            iter += 1

    def benchmark(self):
        '''
        Checking actual solution using scipy
        Must be run before solve.
        '''
        import scipy
        res = scipy.optimize.linprog(self.c, A_ub=self.A, b_ub=self.b)
        print(res.fun)
        print(res.x)
