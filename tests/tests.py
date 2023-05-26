from src.solver import lm, init_values, cor, lambdaCalculation, cvec, LassoShooting_fit
import numpy as np

n = 500
k = 10
k_n0 = 5
X = np.random.normal(0, 1, (n, k))
beta = np.concatenate([np.zeros(k-k_n0), np.ones(k_n0)], axis = 0).reshape((k, 1))
np.random.shuffle(beta)
y = X @ beta + np.random.normal(0, 1, (n, 1))

res = lm(X, y, intercept = False)
init = init_values(X, y, intercept = False)

lmbdas = lambdaCalculation(x = X, y = y)

lshoot = LassoShooting_fit(X, y, lmbdas["lambda"])["coefficients"]