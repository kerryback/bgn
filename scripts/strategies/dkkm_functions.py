import numpy as np 
import pandas as pd 
#from vasicek import *
from joblib import Parallel, delayed
import scipy.linalg as linalg
import statsmodels.api as sm 
from models.parameters import *

# DKKM standardization
def rank_standardize(arr):
    ranks = arr.rank(axis = 0) 
    ranks = (ranks - 0.5) / len(ranks) - 0.5
    return ranks

# random Fourier factor portfolios for a single month
def rff(data, rf, W, model):
    X = rank_standardize(data)
    if model == 'bgn':
        X['rf'] = rf
    Z = W @ X.T
    Z1 = np.sin(Z)
    Z2 = np.cos(Z)
    arr = pd.concat([Z1, Z2], axis = 0).T
    arr.columns = [str(i) for i in range(arr.shape[1])]
    return rank_standardize(arr) , arr

# calculate fama-macbeth weights for a single month
def fama_macbeth(data, **kwargs):
    d = pd.DataFrame(data)

    # standardize characteristics
    d = d.apply(
        lambda x: x / x.std() if x.std() != 0 else 0, 
        axis=0
    ) 

    # FM portfolios
    X = sm.add_constant(d)
    P = X @ linalg.pinvh(X.T @ X) 
    P = P.iloc[:, 1:]
    P *= 2 / P.abs().sum()
    P.columns = [str(i) for i in range(P.shape[1])]
    return P

# panel of random Fourier factor returns
def factors(panel, W, n_jobs, start, end, model):
    def monthly_rets(month):
        data = panel.loc[month]

        if model == 'bgn':
            rf = (data.rf_stand)
        else:
            rf = None

        weights_rs, weights_nors = rff(data[chars], rf, W=W, model=model)
        return month, (weights_rs.T @ data.xret).astype(np.float32), (weights_nors.T @ data.xret).astype(np.float32)
    lst = Parallel(n_jobs=n_jobs, verbose=0)(
        delayed(monthly_rets)(month) for month in range(start, end+1)
    )
    #lst = [monthly_rets(month) for month in range(start, end+1)]
    
    f_nors = pd.concat([x[2] for x in lst], axis=1).T
    f_rs = pd.concat([x[1] for x in lst], axis=1).T
    f_nors["month"] = [x[0] for x in lst]
    f_nors.sort_values(by="month", inplace=True)
    f_nors.set_index("month", inplace=True)
    f_rs["month"] = [x[0] for x in lst]
    f_rs.sort_values(by="month", inplace=True)
    f_rs.set_index("month", inplace=True)
    return f_rs, f_nors

# DKKM portfolio based on past 360 months of factor returns f
def mve_data(f, month, alpha, mkt_rf = None):

    X = f.loc[month-360:month-1].dropna().to_numpy() 
    include_mkt = mkt_rf is not None

    if include_mkt:
        X = np.column_stack((X, mkt_rf.loc[month-360:month-1].dropna().to_numpy()))
    y = np.ones(len(X))
    
    if include_mkt and alpha > 0:
        # augment for unpenalized last variable
        X = np.concatenate((X, np.sqrt(360 * alpha) * np.eye(X.shape[1])[:-1]), axis=0)
        y = np.concatenate([y, np.zeros((X.shape[1] - 1,))])
        alpha = 0  # no additional penalty after augment

    
    T, P = X.shape
    if T < P:
        try:
            # Preferred approach
            U, d, VT = linalg.svd(X @ X.T)
            
            V = VT.T
            W = X.T @ V @ np.diag(1/np.sqrt(d))
            # sequential multiplication avoids creating PxP matrix
            XTy = X.T @ y
            WTXTy = W.T @ XTy
            pi = W @ np.diag(1 / (d + 360*alpha)) @ WTXTy
        except linalg.LinAlgError:
        
            V, d, VT = linalg.svd(X.T @ X)
            pen = 360*alpha*np.ones_like(d)
            pi = V @ np.diag(1 / (d + pen)) @ VT @ X.T @ y
    else:
        D = np.ones(X.shape[1])
        if include_mkt:
            D[-1] = 0  # no penalty on last coeff
        pen = 360 * alpha * np.diag(D)
        pi = np.linalg.solve(X.T @ X + pen, X.T @ y)

    index_cols = list(f.columns) + (['mkt_rf'] if include_mkt else [])
    return pd.Series(pi, index=index_cols)

