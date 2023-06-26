import joblib as jbl
import multiprocess as mp
import numpy as np
from scipy.stats import norm
import pandas as pd
import warnings

from src.solver import cvec, init_values, lambdaCalculation, LassoShooting_fit, lm

def rlasso(x, y, colnames=None, post=True, intercept=True, 
           model=True, homoskedastic=False, X_dependent_lambda=False, 
           lambda_start=None, c=1.1, gamma=None, numSim=5000, 
           numIter=15, tol=10**(-5), threshold=-np.inf, par=True, 
           corecap=np.inf, fix_seed=True):
    
    if isinstance(x, pd.DataFrame) and colnames is None:
        colnames = x.columns
    
    x = np.array(x).astype(np.float64)
    y = cvec(y).astype(np.float64)

    n, p = x.shape

    if colnames is None:
        colnames = ['V' + str(i + 1) for i in np.arange(p)]
    
    if gamma is None:
        gamma = .1 / np.log(n)

    if (post == None) and (c == None):
        c = .5
    
    if ((not post or homoskedastic or X_dependent_lambda) 
        and (lambda_start == None) and (c == 1.1) 
        and (gamma == .1 / np.log(n))):
        c = .5

    est = None

    if intercept:
        meanx = cvec(x.mean(axis = 0)).reshape((p, 1))
        x = x - meanx.T
        mu = y.mean()
        y = y - mu
        
    else:
        meanx = np.zeros((p, 1))
        mu = 0
    
    normx = np.sqrt(np.var(x, axis = 1, ddof = 1))
    Psi = cvec(np.mean(x ** 2, axis = 0))
    ind = np.zeros((p, 1)).astype(bool)
    XX = x.T @ x
    Xy = x.T @ y
    startingval = init_values(x, y)["residuals"]
    pen = lambdaCalculation(homoskedastic = homoskedastic, 
                            X_dependent_lambda = X_dependent_lambda, 
                            lambda_start = lambda_start, c = c, gamma = gamma, 
                            numSim = numSim, y = startingval, x = x, par = par, 
                            corecap = corecap, fix_seed = fix_seed)
    lmbda = pen["lambda"]
    Ups1 = Ups0 = pen["Ups0"]
    lmbda0 = pen["lambda0"]

    mm = 1
    s0 = np.sqrt(np.var(y, axis = 0, ddof = 1))
    break_cond = False

    while (mm <= numIter) and not break_cond:

        if (mm == 1) and post:
            coefTemp = LassoShooting_fit(x, y, lmbda / 2, XX = XX, 
                                         Xy = Xy)["coefficients"]
            
        else:
            coefTemp = LassoShooting_fit(x, y, lmbda, XX = XX, 
                                         Xy = Xy)["coefficients"]
        
        coefTemp[np.isnan(coefTemp)] = 0
        ind1 = (np.abs(coefTemp) > 0)
        x1 = x[:, ind1[:, 0]]

        if x1.shape[1] == 0:

            if intercept:
                intercept_value = np.mean(y + mu)
                coef = np.zeros((p + 1, 1))
                coef = pd.DataFrame(coef, 
                                    index = ["(Intercept)"] + list(colnames))                
            else:
                intercept_value = np.mean(y)
                coef = np.zeros((p, 1))
                coef = pd.DataFrame(coef, index  = list(colnames))
            
            est = {
                "coefficients": coef, 
                "beta": np.zeros((p, 1)), 
                "intercept": intercept_value,
                "index": pd.DataFrame(np.zeros((p, 1)).astype(bool), 
                                      index = colnames), 
                "lambda": lmbda, 
                "lambda0": lmbda0,
                "loadings": Ups0, 
                "residuals": y - np.mean(y), 
                "sigma": np.var(y, axis = 0, ddof = 1), 
                "iter": mm, 
                "options": {"post": post, "intercept": intercept, 
                            "ind.scale": ind, "mu": mu, "meanx": meanx}
            }

            if model:
                est["model"] = x
            else:
                est["model"] = None
            
            est["tts"] = est["rss"] = (y - np.mean(y) ** 2).sum()
            est["dev"] = y - np.mean(y)
            
            return est
        
        if post:
            reg = lm(x1, y, intercept = False)
            coefT = reg
            coefT[np.isnan(coefT)] = 0
            e1 = y - x1 @ coefT
            coefTemp[ind1[:, 0]] = coefT
        
        else:
            e1 = y - x1 @ coefTemp[ind1[:, 0]]
        
        s1 = np.sqrt(np.var(e1, ddof = 1))

        # Homoskedastic and X-independent or X-dependent

        if homoskedastic:
            Ups1 = s1 * Psi
            lmbda = pen["lambda0"] * Ups1
            
        # Heteroskedastic and X-independent
        elif not X_dependent_lambda:
            Ups1 = (1 / np.sqrt(n)) * np.sqrt((e1 ** 2).T @ x ** 2).T
            lmbda = pen["lambda0"] * Ups1
        
        # Heteroskedastic and X- dependent
        else:
            lc = lambdaCalculation(homoskedastic = homoskedastic, 
                                   X_dependent_lambda = X_dependent_lambda, 
                                   lambda_start = lambda_start, c = c, 
                                   gamma = gamma, numSim = numSim, y = startingval, 
                                   x = x, par = par, corecap = corecap, 
                                   fix_seed = fix_seed)
            Ups1 = lc["Ups0"]
            lmbda = lc["lambda"]
        
        mm += 1

        if x1.shape[1] == 0:
            ind1 = np.zeros((p, 1))
        
        coefTemp = cvec(coefTemp)
        coefTemp[np.abs(coefTemp) < threshold] = 0
        coefTemp = pd.DataFrame(coefTemp, index = colnames)
        ind1 = cvec(ind1)
        ind1 = pd.DataFrame(ind1, index = colnames)

        if intercept:
            
            if mu is None:
                mu = 0
            
            if meanx is None:
                meanx = np.zeros((coefTemp.shape[0], 1))

            if ind.sum() == 0:
                intercept_value = mu - (meanx.T @ coefTemp.values).sum()

            else:
                intercept_value = mu - (meanx.T @ coefTemp.values).sum()
            
            beta = np.concatenate([cvec(intercept_value), coefTemp.values], 
                                  axis = 0)
            beta = pd.DataFrame(beta, index = ["(Intercept)"] + list(colnames))
            
        else:
            intercept_value = np.nan
            beta = coefTemp
        
        s1 = np.sqrt(np.var(e1, ddof = 1))

        break_cond = (np.abs(s1 - s0) < tol)

        s0 = s1

        if mm == numIter:
            warnings.warn("Reached maximum number of iterations")

    est = {
        "coefficients": beta, 
        "beta": pd.DataFrame(coefTemp, index = colnames), 
        "intercept": intercept_value, 
        "index": ind1,
        "lambda": pd.DataFrame(lmbda, index = colnames), 
        "lambda0": lmbda0, 
        "loadings": Ups1, 
        "residuals": cvec(e1), 
        "sigma": s1, 
        "iter": mm, 
        "options": {"post": post, "intercept": intercept, 
                    "ind.scale": ind, "mu": mu, "meanx": meanx}, 
        "model": model
    }

    
    if model:
        x = x + meanx.T
        est["model"] = x
        
    else:
        est["model"] = None
    
    est['tss'] = ((y - np.mean(y)) ** 2).sum()
    est['rss'] = (est['residuals'] ** 2).sum()
    est['dev'] = y - np.mean(y)

    return est

def rlassoIVselectX():
    return

def rlassoIVselectZ(x, d, y, z, post = True, intercept = True, **kwargs):

    d = cvec(d)
    z = cvec(z)
    x = cvec(x)
    y = cvec(y)

    n = y.shape[0]
    kex = x.shape[1]
    ke = d.shape[1]

    if isinstance(d, pd.DataFrame):
        colnames_d = d.columns
    else:
        colnames_d = ["d" + str(i + 1) for i in np.arange(ke)]
    
    if isinstance(x, pd.DataFrame):
        colnames_x = x.columns
    else:
        colnames_x = ["x" + str(i + 1) for i in np.arange(kex)]
    
    Z = np.concatenate([x, z], axis = 1)
    kiv = Z.shape[1]
    select_mat = np.array([[] for i in np.arange(kiv)], dtype = bool)
    Dhat = np.array([[] for i in np.arange(n)])
    flag_const = 0
    
    for i in np.arange(ke):
        di = d[:, [i]]
        lasso_fit = rlasso(y = di, x = Z, post = post, intercept = intercept, **kwargs)
        if not np.any(lasso_fit["index"]):
            dihat = np.repeat((n, 1), np.mean(di))
            flag_const += 1
            if flag_const > 1:
                warnings.warn("No variables selected for two or more instruments, leading to multicollinearity problems")
            select_mat = np.concatenate([select_mat, np.full((kiv, 1), False)], 
                                        axis = 1)
        else:
            dihat = di - lasso_fit["residuals"].to_numpy().reshape((n, 1))
            select_mat = np.concatenate([select_mat, lasso_fit["index"]], axis = 1)
        
        Dhat = np.concatenate([Dhat, dihat], axis = 1)
    
    Dhat = np.concatenate([Dhat, x], axis = 1)
    d = np.concatenate([d, x], axis = 1)
    alpha_hat = np.linalg.pinv(Dhat.T @ d) @ (Dhat.T @ y)
    residuals = y - d @ alpha_hat
    Omega_hat = Dhat.T @ np.diag((residuals ** 2).reshape(n)) @ Dhat
    Q_hat_inv = np.linalg.ping(d.T @ Dhat)
    vcov = Q_hat_inv @ Omega_hat @ Q_hat_inv.T
    

        
        



