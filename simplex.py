import numpy as np
import enum

np.set_printoptions(linewidth=np.inf, suppress=True)

# TODO: replace for loops with list comprehension


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
        ''' Add slack variables
        '''
        self.n += self.m
        self.c = np.append(self.c, np.zeros(self.m))
        self.A = np.append(self.A, np.eye(self.m), axis=1)

    def tabularize(self):
        ''' Convert to tableau format
        '''
        # initial basis
        basis = [np.argmax((self.A.T == row).all(axis=1))
                 for row in np.eye(self.m)]
        self.basic_var = np.array([basis])

        obj_row = np.append(-self.c, [0])
        self.table = np.append(self.A, np.array([self.b]).T, axis=1)

        # zero in objective row for basic variables
        for i in range(self.m):
            obj_row -= obj_row[basis[i]] * self.table[i]

        self.table = np.append([obj_row], self.table, axis=0)

    def get_solution(self):
        ''' Solution from tableau
        '''
        x = np.zeros(self.n)
        x[self.basic_var[-1]] = self.table.T[-1][1:]
        return self.table[0][-1], x

    def output(self):
        if self.status == self.Status.OPTIMAL:
            res, x_optim = self.get_solution()
            print(res)
            print(*x_optim[:self.n-self.m])
        elif self.status == self.Status.UNBOUNDED:
            print("Unbounded")
        elif self.status == self.Status.INFEASIBLE:
            print("Infeasible")

    def detect_cycle(self, arr, ele):
        if self.blands_rule:
            return
        if np.any([set(x) == set(ele) for x in arr]):
            self.blands_rule = True

    def iteration(self):
        basis = self.basic_var[-1].copy()
        if self.debug:
            print('Basis', basis)

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

        self.table = self.pivot(
            self.table.copy(), leaving_var + 1, entering_var)

        basis[leaving_var] = entering_var
        self.detect_cycle(self.basic_var, basis)
        self.basic_var = np.append(self.basic_var, [basis], axis=0)
        return

    def pivot(self, table, r, c):
        ''' make all elements in col c as 0 except for row r as 1
        '''
        assert table[r][c] != 0, "invalid pivot"

        table[r] /= table[r][c]

        # Pivoting operations
        for i in range(len(table)):
            if i != r:
                coeff = table[i][c]
                table[i] -= coeff * table[r]

        return table

    def two_phase(self):
        # introduce artifical variables
        art = (self.b < 0)
        A = self.A * (1-2*art)[:, None]
        b = self.b * (1-2*art)

        art = np.eye(self.m) * art
        art = art[~np.all(art == 0, axis=1)].T
        A = np.append(A, art, axis=1)
        c = np.append(np.zeros(self.n), np.ones(art.shape[1]))

        phase1 = Simplex(A, b, c)
        phase1.solve_tableau()

        res, _ = phase1.get_solution()
        if res > 0:
            self.status = self.Status.INFEASIBLE
            return

        table = np.append(
            phase1.table[:, :-1-art.shape[1]], phase1.table[:, -1][:, None], axis=1)[1:]

        # remove artificial variables from basis
        for i, bs in enumerate(phase1.basic_var[-1]):
            if (bs >= table.shape[1]-1) and np.any(table[i][:-1] != 0):
                j = np.argmax(table[i][:-1] != 0)
                table = self.pivot(table.copy(), i, j)

        for i, row in enumerate(table):
            # eliminate zero rows
            if np.all(row[:-1] == 0):
                table = np.append(table[:i], table[i+1:], axis=0)

        self.A = table[:, :-1]
        self.b = table[:, -1]

    def solve(self):
        self.standardize()

        if np.any(self.b < 0):
            self.two_phase()

        if self.status == self.Status.SOLVING:
            self.solve_tableau()

    def solve_tableau(self):
        '''
        Same as solve but for constraint of form Ax = b with feasible solution.
        '''
        self.tabularize()

        iter = 1
        while self.status == self.Status.SOLVING:
            if self.debug:
                print('Iteration:', iter)
                self.visualize()
            self.iteration()
            if self.debug:
                print('----')
            iter += 1

    # COMMENT BEFORE SUBMIT
    def benchmark(self):
        '''
        Checking actual solution using scipy
        Must be run before solve.
        '''
        import scipy
        res = scipy.optimize.linprog(self.c, A_ub=self.A, b_ub=self.b)
        print(res.fun)
        print(res.x)
