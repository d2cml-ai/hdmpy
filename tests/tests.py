import numpy as np
import pandas as pd
from patsy import dmatrix

from src.solver import lm, init_values, cor, lambdaCalculation, cvec, LassoShooting_fit
from src.estimation import rlasso
from src.data import *

# -------- Randomly Generated Example -------- #

n = 500
k = 15
k_n0 = 5
X = np.random.normal(0, 1, (n, k))
beta = np.concatenate([np.zeros(k-k_n0), np.ones(k_n0) * 5], axis = 0).reshape((k, 1))
np.random.shuffle(beta)
y = X @ beta + np.random.normal(0, 1, (n, 1))

# res = lm(X, y, intercept = False)
# init = init_values(X, y, intercept = False)

# lmbdas = lambdaCalculation(x = X, y = y)

# lshoot = LassoShooting_fit(X, y, lmbdas["lambda"])["coefficients"]

res = rlasso(X, y, post = False)

# -------- Institutional Quality -------- #

AJR = load_AJR()
n = AJR.shape[0]

y = AJR["GDP"].to_numpy().reshape((n, 1))
d = AJR["Exprop"].to_numpy().reshape((n, 1))
z = AJR["logMort"].to_numpy().reshape((n, 1))
X = dmatrix("-1 + (Latitude + Latitude2 + Africa + Asia + Namer + Samer) ** 2", AJR)[:]

corr = cor(X, y)

index = corr.argsort()[-np.amin([5, 21]):]


AJR_rl = rlasso(X, y)


# -------- Gender Wage Gap Example -------- #

data = load_cps2012()