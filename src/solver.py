import joblib as jbl
import multiprocess as mp
import numpy as np
from scipy.stats import norm
import pandas as pd
import numbers


def cvec(x):

    if isinstance(x, numbers.Number):
        x = np.array([[x]])
    elif len(x.shape) < 2:
        x.reshape((x.shape[0], 1))
    
    return x

def lm(X, y, intercept = True):
    """ An OLS regression of y on X

    Inputs
    y: n by 1 NumPy array
    X: n by k NumPy array

    Outputs
    coefs: k by 1 NumPy array with regression coefficients
    """

    assert X.shape[0] == y.shape[0], "row numbers differ b/w X and y"
    assert (len(X.shape), len(y.shape)) == (2, 2), "X and y must be 2d arrays"

    if intercept:
        X = np.concatenate([np.ones((X.shape[0], 1)), X], axis = 1)

    XX = X.T @ X
    XX_inv = np.linalg.inv(XX)
    Xy = X.T @ y
    coefs = XX_inv @ Xy

    if intercept:
        return coefs[1:, :]
    
    return coefs

def cor(y, X):
    """ Return correlation coefficients between columns of matrices

    Inputs
    y: n by 1 NumPy array
    X: n by k NumPy array

    Outputs
    corr: list of length k, where the k-th element is the correlation
          coefficient between y and the k-th column of X
    """
    rand_vars = np.where(np.var(X, axis = 0) != 0)[0]
    
    # Concatenate y and X into a single NumPy array
    yX = np.concatenate([y, X[:, rand_vars]], axis=1)

    # Get the correlation coefficients between all columns of that array
    corr = np.zeros(X.shape[1])
    corr[rand_vars] = np.corrcoef(yX, rowvar=False)[0, 1:]

    # Get the first row, starting at the first off-diagonal element (these are
    # the correlation coefficients between y and each column of X
    corr
    corr[np.isnan(corr)] = 0

    # Return the result
    return corr

def init_values(X, y, number=5, intercept=True):
    """ Return an initial parameter guess for a LASSO model

    Inputs
    y: n by 1 NumPy array, outcome variable
    X: n by k NumPy array, RHS variables

    Outputs
    residuals: n ny 1 NumPy array, residuals for initial parameter guess
    coefficients: k by 1 NumPy array, initial coefficient values
    """
    # Make sure y is a proper column vector
    cvec(y)

    # Get the absolute value of correlations between y and X
    corr = np.abs(cor(y, X))

    # Get the number of columns of X
    kx = X.shape[1]

    # Make an index selecting the five columns of X which are most correlated
    # with y (since .argsort() always sorts in increasing order, selecting from
    # the back gets the most highly correlated columns)
    index = corr.argsort()[-np.amin([number, kx]):]

    # Set up an array of coefficient guesses
    coefficients = np.zeros((kx, 1))

    # Regress y on the five most correlated columns of X, including an intercept
    # if desired
    reg = lm(X[:, index], y, intercept=intercept)

    # Replace the guesses for the estimated coefficients (note that .coef_ does
    # not return the estimated intercept, if one was included in the model)
    coefficients[index, :] = reg

    # Replace any NANs as zeros
    coefficients[np.isnan(coefficients)] = 0

    # Get the regression residuals
    residuals = y - X[:, index] @ reg

    # Return the residuals and coefficients
    return {'residuals': residuals, 'coefficients': coefficients}


def LassoShooting_fit(x, y, lmbda, maxIter = 1000, optTol = 1e-5, zeroThreshold = 1e-6, XX = None, Xy = None, beta_start = None):

    y = cvec(y)
    lmbda = cvec(lmbda)

    n, p = x.shape

    # Check whether XX and Xy were provided, calculate them if not
    if XX is None:
        XX = x.T @ x
    if Xy is None:
        Xy = x.T @ y

    # Check whether an initial value for the intercept was provided
    if beta_start is None:
        # If not, use init_values from help_functions, which will return
        # regression estimates for the five variables in x which are most
        # correlated with y, and initialize all other coefficients as zero
        beta = init_values(x, y, intercept=False)["coefficients"]
    else:
        # Otherwise, use the provided initial weights
        beta = beta_start

    # Set up a history of weights over time, starting with the initial ones
    wp = beta

    # Keep track of the number of iterations
    m = 1

    # Create versions of XX and Xy which are just those matrices times two
    XX2 = XX * 2
    Xy2 = Xy * 2

    # Go through all iterations
    while m < maxIter:
        # Save the last set of weights (the .copy() is important, otherwise
        # beta_old will be updated every time beta is changed during the
        # following loop)
        beta_old = beta.copy()

        # Go through all parameters
        for j in np.arange(p):
            # Calculate the shoot
            S0 = XX2[j,:] @ beta - XX2[j,j] * beta[j,0] - Xy2[j,0]

            # Update the weights
            if np.isnan(S0).sum() >= 1:
                beta[j] = 0
            elif S0 > lmbda[j]:
                beta[j] = (lmbda[j] - S0) / XX2[j,j]
            elif S0 < -lmbda[j]:
                beta[j] = (-lmbda[j] - S0) / XX2[j,j]
            elif np.abs(S0) <= lmbda[j]:
                beta[j] = 0

        # Add the updated weights to the history of weights
        wp = np.concatenate([wp, beta], axis=1)

        # Check whether the weights are within tolerance
        if np.abs(beta - beta_old).sum() < optTol:
            # If so, break the while loop
            break

        # Increase the iteration counter
        m += 1

    # Set the final weights to the last updated weights
    w = beta

    # Set weights which are within zeroThreshold to zero
    w[np.abs(w) < zeroThreshold] = 0

    # Return the weights, history of weights, and iteration counter
    return {'coefficients': w, 'coef.list': wp, 'num.it': m}

def simul_pen(n, p, W, seed=0, fix_seed=True):

    # Check whether the seed needs to be fixed
    if fix_seed:
        # Simulate with provided seed
        g = norm.rvs(size=(n,1), random_state=seed) @ np.ones(shape=(1,p))
    else:
        # Simulate using whatever state the RNG is currently in
        g = norm.rvs(size=(n,1)) @ np.ones(shape=(1,p))

    # Calculate element of the distribution for the current draw of g
    s = n * np.amax(2 * np.abs(np.mean(W * g, axis=0)))

    # Return the result
    return s

def lambdaCalculation(homoskedastic=False, X_dependent_lambda=False,
                      lambda_start=None, c=1.1, gamma=0.1, numSim=5000, y=None,
                      x=None, par=True, corecap=np.inf, fix_seed=True):
    
    # Get number of observations n and number of variables p
    n, p = x.shape

    # Get number of simulations to use (if simulations are necessary)
    R = numSim

    # Go through all possible combinations of homoskedasticy/heteroskedasticity
    # and X-dependent or independent error terms. The first two cases are
    # special cases: Handling the case there homoskedastic was set to None, and
    # where lambda_start was provided.
    #
    # 1) If homoskedastic was set to None (special case)
    if homoskedastic is None:
        # Initialize lambda
        lmbda0 = lambda_start

        Ups0 = (1 / np.sqrt(n)) * np.sqrt((y**2).T @ (x**2)).T

        # Calculate the final vector of penalty terms
        lmbda = lmbda0 * Ups0

    # 2) If lambda_start was provided (special case)
    elif lambda_start is not None:
        # Check whether a homogeneous penalty term was provided (a scalar)
        if np.amax(cvec(lambda_start).shape) == 1:
            # If so, repeat that p times as the penalty term
            lmbda = np.ones(shape=(p,1)) * lambda_start
        else:
            # Otherwise, use the provided vector of penalty terms as is
            lmbda = lambda_start

    # 3) Homoskedastic and X-independent
    elif (homoskedastic == True) and (X_dependent_lambda == False):
        # Initilaize lambda
        lmbda0 = 2 * c * np.sqrt(n) * norm.ppf(1 - gamma/(2*p))

        # Use ddof=1 to be consistent with R's var() function
        Ups0 = np.sqrt(np.var(y, axis=0, ddof=1))

        # Calculate the final vector of penalty terms
        lmbda = np.zeros(shape=(p,1)) + lmbda0 * Ups0

    # 4) Homoskedastic and X-dependent
    elif (homoskedastic == True) and (X_dependent_lambda == True):
        psi = cvec((x**2).mean(axis=0))

        tXtpsi = (x.T / np.sqrt(psi)).T

        # Check whether to use parallel processing
        if par == True:
            # If so, get the number of cores to use
            cores = np.int(np.amin([mp.cpu_count(), corecap]))
        else:
            # Otherwise, use only one core (i.e. run sequentially)
            cores = 1

        # Get simulated distribution
        sim = jbl.Parallel(n_jobs=cores)(
            jbl.delayed(simul_pen)(
                n, p, tXtpsi, seed=l*20, fix_seed=fix_seed
            ) for l in np.arange(R)
        )

        # Convert it to a proper column vector
        sim = cvec(sim)

        # Initialize lambda based on the simulated quantiles
        lmbda0 = c * np.quantile(sim, q=1-gamma, axis=0)

        Ups0 = np.sqrt(np.var(y, axis=0, ddof=1))

        # Calculate the final vector of penalty terms
        lmbda = np.zeros(shape=(p,1)) + lmbda0 * Ups0

    # 5) Heteroskedastic and X-independent
    elif (homoskedastic == False) and (X_dependent_lambda == False):
        # The original includes the comment, "1=num endogenous variables"
        lmbda0 = 2 * c * np.sqrt(n) * norm.ppf(1 - gamma/(2*p*1))

        Ups0 = (1 / np.sqrt(n)) * np.sqrt((y**2).T @ (x**2)).T

        lmbda = lmbda0 * Ups0

    # 6) Heteroskedastic and X-dependent
    elif (homoskedastic == False) and (X_dependent_lambda == True):
        eh = y

        ehat = eh @ np.ones(shape=(1,p))

        xehat = x * ehat

        psi = cvec((xehat**2).mean(axis=0)).T

        tXehattpsi = (xehat / ( np.ones(shape=(n,1)) @ np.sqrt(psi) ))

        # Check whether to use parallel processing
        if par == True:
            # If so, get the number of cores to use
            cores = np.int(np.amin([mp.cpu_count(), corecap]))
        else:
            # Otherwise, use only one core (i.e. run sequentially)
            cores = 1

        # Get simulated distribution
        sim = jbl.Parallel(n_jobs=cores)(
            jbl.delayed(simul_pen)(
                n, p, tXehattpsi, seed=l*20, fix_seed=fix_seed
            ) for l in np.arange(R)
        )

        # Convert it to a proper column vector
        sim = cvec(sim)

        # Initialize lambda based on the simulated quantiles
        lmbda0 = c * np.quantile(sim, q=1-gamma, axis=0)

        Ups0 = (1 / np.sqrt(n)) * np.sqrt((y**2).T @ (x**2)).T

        # Calculate the final vector of penalty terms
        lmbda = lmbda0 * Ups0

    # Return results
    return {'lambda0': lmbda0, 'lambda': lmbda, 'Ups0': Ups0}


    
