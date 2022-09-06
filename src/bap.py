#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse
from scipy.ndimage import convolve1d
from scipy.optimize import minimize

# Domain 
L = 30
dx = 0.1
x = np.arange(-L, L + dx, dx)
N = x.shape[0]

# Problem parameters
eps = 5
alpha = 8
c0 = 0.0625
p = int(np.round(eps/dx))

# Interaction kernel
x_ker = np.arange(-10, 10+dx, dx)
kappa = 0.5*np.exp(-np.abs(x_ker))

# Initial conditions
x_u = -5
x_v = 5
sigma = 20
m = 1

u0 = 1/(sigma*np.sqrt(2*np.pi))*np.exp(-0.5*((x - x_u)/sigma)**2)
u0 = c0 + m/2*(u0/(np.sum(u0)*dx))

v0 = 1/(sigma*np.sqrt(2*np.pi))*np.exp(-0.5*((x - x_v)/sigma)**2)
v0 = c0 + m/2*(v0/(np.sum(v0)*dx))

y0 = np.concatenate((u0, v0))

def tau_e_plus(y):
    return np.roll(y, p)

def tau_e_minus(y):
    return np.roll(y, -p)

def convolve_with_kappa(y):
    return dx*convolve1d(y, kappa, mode='wrap')

def plot_solution(u, v):
    tails = u + v
    heads = tau_e_minus(u) + tau_e_plus(v)
    water = 1 - heads - tails
    plt.plot(x, tails, x, heads, x, water)
    plt.legend(('tails', 'heads', 'water'))

def objective_fun(y):
    u = y[:N]
    v = y[N:]

    um = u > 0
    vm = v > 0
    u_ent = np.zeros(N)
    v_ent = np.zeros(N)

    u_ent[um] = u[um]*np.log(u[um])
    v_ent[vm] = v[vm]*np.log(v[vm])

    return dx*np.sum(u_ent + v_ent + 
            alpha*(1 - u - v) * convolve_with_kappa(u + v))

def nzlog(x):
    y = np.full(x.shape, -1000)
    y[x > 0] = np.log(x[x > 0])
    return y

def grad_fun(y):
    u = y[:N]
    v = y[N:]

    
    foo = alpha*convolve_with_kappa(1 - 2*u - 2*v)
    return np.concatenate((1 + nzlog(u) + foo, 1 + nzlog(v) + foo))

# Equality constraint
A_eq = np.ones((1, 2*N))
rhs_eq = m/dx + 2*N*c0

def fun_eq(y):
    return A_eq.dot(y) - rhs_eq
c_eq = {'type':'eq', 'fun':fun_eq}
c_eq_obj = scipy.optimize.LinearConstraint(A_eq, lb=rhs_eq, ub=rhs_eq)


# Inequality constraint
w = np.arange(N)
i = np.tile(w, (4, 1)).T
j = np.vstack((w, np.mod(w+p, N), w+N, np.mod(w-p, N)+N)).T
data = np.ones(i.shape)
A_ineq = scipy.sparse.coo_matrix((data.flatten(), (i.flatten(),
    j.flatten())), shape=(N, 2*N)).tocsr()
def fun_ineq(y):
    return 1 - A_ineq.dot(y);
c_ineq = {'type':'ineq', 'fun':fun_ineq}
c_ineq_obj = scipy.optimize.LinearConstraint(A_ineq, -np.inf, 1)


# Non-negative constraint
def fun_nn(y):
    return y
c_nn = {'type':'ineq', 'fun':fun_nn}
c_nn_obj = scipy.optimize.LinearConstraint(scipy.sparse.eye(2*N), 0, np.inf)

constraints_sqspy = (c_eq, c_ineq, c_nn)
constraints_trust = (c_eq_obj, c_ineq_obj, c_nn_obj)

res = minimize(objective_fun, y0, method='trust-constr',
        constraints=constraints_trust)

y = res.x
u = y[:N]
v = y[N:]


plt.figure()
plot_solution(u, v)
plt.show()
