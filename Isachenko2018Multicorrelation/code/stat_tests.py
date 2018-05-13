import numpy as np
import scipy as sc


def t_test_corr(r, m, alpha=0.05):
    t = r * np.sqrt(m - 2) / (1 - r ** 2)
    tb = sc.stats.t.ppf(1 - alpha, df=m - 2)
    return tb > np.abs(t)


def check_correlation(X, y):
    m, n = X.shape
    rel = t_test_corr(corr(X, y), m)
    return rel