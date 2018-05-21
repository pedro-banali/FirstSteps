import numpy as np
import pandas as pd
from patsy import dmatrices
import warnings


def sigmoid(x):
    return 1/(1+np.exp(-x))


def catch_singularity(f):
    def silencer(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except np.linalg.LinAlgError:
            warnings.warn('Algorithm terminated - singular Hessian!')
            return args[0]
        return silencer


np.random.seed(0)
tol = 1e-8
lam = None
max_iter = 20

r = 0.95  # covariance between x and z
n = 1000  # number of observations
sigma = 1  # variance of noise

beta_x, beta_z, beta_v = -4, .9, 1
var_x, var_z, var_v = 1, 1, 4

formula = 'y ~ x + z + v + np.exp(x) + I(v**2 + z)'

x, z = np.random.multivariate_normal([0, 0], [[var_x, r], [r, var_z]], n).T
v = np.random.normal(0, var_v, n)**3
A = pd.DataFrame({'x': x, 'z': z, 'v': v})
A['log_odds'] = sigmoid(A[['x', 'z', 'v']].dot([beta_x, beta_z, beta_v]) + sigma * np.random.normal(0, 1, n))
A['y'] = [np.random.binomial(1, p) for p in A.log_odds]

y, X = dmatrices(formula, A, return_type='dataframe')

X.head( )


def newton_step(curr, X, lam=None):
    p = np.array(sigmoid(X.dot(curr[: , 0])), ndmin=2).T
    W = np.diag((p*(1-p))[:, 0])
    hessian = x.T.dot(W).dot(X)
    grad = x.T.dot(y-p)

    if lam:
        step, *_ = np.linalg.lstsq(hessian + lam * np.eye(curr.shape[0]), grad)
    else:
        step, *_ = np.linalg.lstsq(hessian, grad)

    # if lam:
    #     step = np.dot(np.linalg.inv(hessian + lam * np.eye(curr.shape[0])), grad)
    # else:
    #     step = np.dot(np.linalg.inv(hessian), grad)

    beta = curr + step

    return beta


def new_newton_step(curr, X, lam=None):
    p = np.array(sigmoid(X.dot(curr[: , 0])), ndmin=2).T
    W = np.diag((p*(1-p))[:, 0])
    hessian = x.T.dot(W).dot(X)
    grad = x.T.dot(y-p)

    if lam:
        step = np.dot(np.linalg.inv(hessian + lam * np.eye(curr.shape[0])), grad)
    else:
        step = np.dot(np.linalg.inv(hessian), grad)

    beta = curr + step

    return beta


def check_coefs_convergence(beta_old, beta_new, tol, iters):
    coef_change = np.abs(beta_old - beta_new)
    return not (np.any(coef_change > tol) & iters < max_iter)


beta_old, beta = np.ones((len(X.columns), 1)), np.zeros((len(X.columns), 1))

iter_count = 0
coefs_converged = False

while not coefs_converged:
    beta_old = beta
    beta = newton_step(beta, X, lam=lam)

    iter_count += 1

    coefs_converged = check_coefs_convergence(beta_old, beta, tol, iter_count)


print('Iterations : {}'.format(iter_count))
print('Beta: {}'.format(beta))
