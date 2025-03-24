import os

from devito.symbolics import retrieve_functions, INT
import pandas as pd
from matplotlib import pyplot as plt
from devito import configuration, Operator, Eq, Grid, Function
from devito.petsc import PETScSolve, EssentialBC
from devito.petsc.initialize import PetscInitialize
configuration['opt'] = 'noop'
configuration['compiler'] = 'custom'
os.environ['CC'] = 'mpicc'

# Modified Helmholtz equation

PetscInitialize()


Lx = 1.
Ly = Lx

# Number of grid points in each direction
n = 11

# mesh spacing
dx = Lx/(n-1)
dy = Ly/(n-1)
so = 2


class SubTop(SubDomain):
    name = 'subtop'
    def __init__(self, S_O):
        super().__init__()
        self.S_O = S_O

    def define(self, dimensions):
        x, y = dimensions
        return {x: ('middle', 1, 1), y: ('right', self.S_O//2)}
sub1 = SubTop(so)

class SubBottom(SubDomain):
    name = 'subbottom'
    def __init__(self, S_O):
        super().__init__()
        self.S_O = S_O

    def define(self, dimensions):
        x, y = dimensions
        return {x: ('middle', 1, 1), y: ('left', self.S_O//2)}
sub2 = SubBottom(so)

class SubLeft(SubDomain):
    name = 'subleft'
    def __init__(self, S_O):
        super().__init__()
        self.S_O = S_O

    def define(self, dimensions):
        x, y = dimensions
        return {x: ('left', self.S_O//2), y: ('middle', 1, 1)}
sub3 = SubLeft(so)

class SubRight(SubDomain):
    name = 'subright'
    def __init__(self, S_O):
        super().__init__()
        self.S_O = S_O

    def define(self, dimensions):
        x, y = dimensions
        return {x: ('right', self.S_O//2), y: ('middle', 1, 1)}
sub4 = SubRight(so)


class SubPointBottomLeft(SubDomain):
    name = 'subpointbottomleft'
    def define(self, dimensions):
        x, y = dimensions
        return {x: ('left', 1), y: ('left', 1)}
sub5 = SubPointBottomLeft()

class SubPointBottomRight(SubDomain):
    name = 'subpointbottomright'
    def define(self, dimensions):
        x, y = dimensions
        return {x: ('right', 1), y: ('left', 1)}
sub6 = SubPointBottomRight()

class SubPointTopLeft(SubDomain):
    name = 'subpointtopleft'
    def define(self, dimensions):
        x, y = dimensions
        return {x: ('left', 1), y: ('right', 1)}
sub7 = SubPointTopLeft()

class SubPointTopRight(SubDomain):
    name = 'subpointtopright'
    def define(self, dimensions):
        x, y = dimensions
        return {x: ('right', 1), y: ('right', 1)}
sub8 = SubPointTopRight()


def neumann_bottom(eq, subdomain):
    lhs, rhs = eq.evaluate.args

    # Get vertical subdimension and its parent
    yfs = subdomain.dimensions[-1]
    y = yfs.parent

    # Functions present in stencil
    funcs = retrieve_functions(lhs-rhs)

    mapper = {}
    for f in funcs:
        # Get the y index
        yind = f.indices[-1]
        if (yind - y).as_coeff_Mul()[0] < 0:
            mapper.update({f: f.subs({yind: INT(abs(yind))})})

    return Eq(lhs.subs(mapper), rhs.subs(mapper), subdomain=subdomain)


def neumann_top(eq, subdomain):
    lhs, rhs = eq.evaluate.args

    # Get vertical subdimension and its parent
    yfs = subdomain.dimensions[-1]
    y = yfs.parent

    # Functions present in stencil
    funcs = retrieve_functions(lhs-rhs)

    mapper = {}
    for f in funcs:
        # Get the y index
        yind = f.indices[-1]
        if (yind - y).as_coeff_Mul()[0] > 0:
            # Symmetric mirror
            tmp = y - INT(abs(y.symbolic_max - yind))
            mapper.update({f: f.subs({yind: tmp})})

    return Eq(lhs.subs(mapper), rhs.subs(mapper), subdomain=subdomain)


def neumann_left(eq, subdomain):
    lhs, rhs = eq.evaluate.args

    # Get horizontal subdimension and its parent
    xfs = subdomain.dimensions[0]
    x = xfs.parent

    # Functions present in stencil
    funcs = retrieve_functions(lhs-rhs)

    mapper = {}
    for f in funcs:
        # Get the x index
        xind = f.indices[-2]
        if (xind - x).as_coeff_Mul()[0] < 0:
            # Symmetric mirror
            # Substitute where index is negative for +ve where index is positive
            mapper.update({f: f.subs({xind: INT(abs(xind))})})

    return Eq(lhs.subs(mapper), rhs.subs(mapper), subdomain=subdomain)


def neumann_right(eq, subdomain):
    lhs, rhs = eq.evaluate.args

    # Get horizontal subdimension and its parent
    xfs = subdomain.dimensions[0]
    x = xfs.parent

    # Functions present in stencil
    funcs = retrieve_functions(lhs-rhs)

    mapper = {}
    for f in funcs:
        # Get the x index
        xind = f.indices[-2]
        if (xind - x).as_coeff_Mul()[0] > 0:
            tmp = x - INT(abs(x.symbolic_max - xind))
            mapper.update({f: f.subs({xind: tmp})})

    return Eq(lhs.subs(mapper), rhs.subs(mapper), subdomain=subdomain)


def analytical_solution(x, y):
    return np.cos(2*np.pi*x)*np.cos(2*np.pi*y)


n_values = [11, 21, 31, 41, 51, 61, 71, 81, 91, 101]
h = np.array([Lx/(n-1) for n in n_values])
errors = []


for n in n_values:
    grid = Grid(shape=(n, n), extent=(Lx, Ly), subdomains=(sub1,sub2,sub3,sub4,sub5,sub6,sub7,sub8), dtype=np.float64)
    time = grid.time_dim
    t = grid.stepping_dim
    x, y = grid.dimensions

    u = Function(name='u', grid=grid, space_order=so, dtype=np.float64)
    f = Function(name='f', grid=grid, space_order=so, dtype=np.float64)

    tmpx = np.linspace(0, Lx, n).astype(np.float64)
    tmpy = np.linspace(0, Ly, n).astype(np.float64)
    Y, X = np.meshgrid(tmpx, tmpy)
    f.data[:] = (1.+(8.*(np.pi**2)))*np.cos(2.*np.pi*X)*np.cos(2.*np.pi*Y)

    eqn = Eq(-u.laplace+u, f, subdomain=grid.interior)

    bcs = [neumann_top(eqn, sub1)]
    bcs += [neumann_bottom(eqn, sub2)]
    bcs += [neumann_left(eqn, sub3)]
    bcs += [neumann_right(eqn, sub4)]
    bcs += [neumann_left(neumann_bottom(eqn, sub5), sub5)] # left bottom
    bcs += [neumann_right(neumann_bottom(eqn, sub6), sub6)] # right bottom
    bcs += [neumann_left(neumann_top(eqn, sub7), sub7)] # left top
    bcs += [neumann_right(neumann_top(eqn, sub8), sub8)] # right top

    solver = PETScSolve([eqn]+bcs, target=u, solver_parameters={'rtol': 1e-8})

    with switchconfig(openmp=False):
        op = Operator(solver)
        op.apply()

    analytical_soln = analytical_solution(X, Y)

    error =  np.linalg.norm(analytical_soln[:] - u.data[:]) / np.linalg.norm(analytical_soln[:])
    errors.append(error)

print(errors)
slope, _ = np.polyfit(np.log(h), np.log(errors), 1)
print(slope)

assert slope > 1.9
assert slope < 2.1


# Plot convergence
plt.figure(figsize=(8, 6))
plt.loglog(dx, errors, 'o-', label=f'Error (Slope: {slope:.2f})')
plt.xlabel("h")
plt.ylabel("Error")
plt.title("Convergence Plot: Error vs. dx")
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.legend()
plt.savefig("Helmholtz_convergence.png", dpi=300)
plt.show()