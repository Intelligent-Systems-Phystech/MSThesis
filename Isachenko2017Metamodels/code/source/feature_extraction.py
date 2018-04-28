import numpy as np
import pandas as pd
import scipy as sc
from sklearn.linear_model import LinearRegression
from scipy.interpolate import splrep, splev, splprep


def get_expert_names(params):
    """
    Function gives feature names for expert features
    Input:
    - params - list of hyperparametes
    Output:
    - feature names - list of feature names
    """

    feature_names = ['avg_x', 'avg_y', 'avg_z',
                     'std_x', 'std_y', 'std_z',
                     'abs_x', 'abs_y', 'abs_z', 'mean']
    for i in range(10):
        name = str(i) + '_'
        feature_names += [name + 'x', name + 'y', name + 'z']

    return feature_names


def get_expert_features(ts, params):
    """
    Function returns extracted features
    Input:
    - ts - time series with three axis
    - params - list of hyperparametes
    Output:
    - features - list with extracted features
    """

    x = ts[0]
    y = ts[1]
    z = ts[2]
    n = x.shape[0]
    features = []
    features.append(x.mean())
    features.append(y.mean())
    features.append(z.mean())
    features.append(x.std())
    features.append(y.std())
    features.append(z.std())
    features.append(np.abs(x - x.mean()).mean())
    features.append(np.abs(y - y.mean()).mean())
    features.append(np.abs(z - z.mean()).mean())
    features.append((x+y+z).mean() / 3.)
    x_range = np.linspace(x.min(), x.max(), 11)
    y_range = np.linspace(y.min(), y.max(), 11)
    z_range = np.linspace(z.min(), z.max(), 11)
    for i in range(10):
        features.append(1. * np.sum((x_range[i] <= x) &
                                    (x < x_range[i+1])) / n)
        features.append(1. * np.sum((y_range[i] <= y) &
                                    (y < y_range[i+1])) / n)
        features.append(1. * np.sum((z_range[i] <= z) &
                                    (z < z_range[i+1])) / n)

    return features


def get_autoregressive_names(params):
    """
    Function gives feature names for autoregression model features
    Input:
    - params - list of hyperparametes
    Output:
    - feature names - list of feature names
    """

    n = params[0]
    feature_names = []
    for ax in ['x', 'y', 'z']:
        feature_names += ['intercept_' + ax]
        for i in range(n):
            feature_names += ['coef_' + str(i) + '_' + ax]

    return feature_names


def get_autoregressive_features(ts, params):
    """
    Function returns extracted features
    Input:
    - ts - time series with three axis
    - params - list of hyperparametes
    Output:
    - features - list with extracted features
    """

    n = params[0]
    x = ts[0]
    y = ts[1]
    z = ts[2]
    m = x.shape[0]
    features = []
    X = np.zeros([m-n, n])
    Y = np.zeros(m-n)
    for axis in [x, y, z]:
        for i in range(m-n):
            X[i, :] = axis[i:i+n]
            Y[i] = axis[i+n]
        lr = LinearRegression()
        lr.fit(X, Y)
        features.append(lr.intercept_)
        features.extend(lr.coef_)

    return features


def get_spectrum_names(params):
    """
    Function gives feature names for spectrum features
    Input:
    - params - list of hyperparametes
    Output:
    - feature names - list of feature names
    """

    n = params[0]
    feature_names = []
    for ax in ['x', 'y', 'z']:
        for i in range(n):
            feature_names += ['eigv_' + str(i) + '_' + ax]

    return feature_names


def get_spectrum_features(ts, params):
    """
    Function returns extracted features
    Input:
    - ts - time series with three axis
    - params - list of hyperparametes
    Output:
    - features - list with extracted features
    """

    n = params[0]
    x = ts[0]
    y = ts[1]
    z = ts[2]
    m = x.shape[0]
    features = []
    X = np.zeros([m-n, n])
    Y = np.zeros(m-n)
    for axis in [x, y, z]:
        for i in range(m-n):
            X[i, :] = axis[i:i+n]
        h = sc.linalg.svd(X.T.dot(X), compute_uv=False, overwrite_a=True)
        features.extend(h)

    return features


def get_spline_names(params):
    """
    Function gives feature names for splines features
    Input:
    - params - list of hyperparametes
    Output:
    - feature names - list of feature names
    """

    n = params[0]
    feature_names = []
    for ax in ['x', 'y', 'z']:
        for i in range(n):
            feature_names += ['coef_' + str(i) + '_' + ax]

    return feature_names


def get_spline(t, ts, n):

    s_down = 1e-6
    s_up = 1000.
    spl = splrep(t, ts, s=s_up)
    while len(spl[1]) >= n:
        spl = splrep(t, ts, s=s_up)
        s_up *= 2.
    max_iter = int(np.floor(np.log2(s_up * 1e4)))
    num_iter = 0
    while (len(spl[1]) != n) and (num_iter <= max_iter):
        s = (s_up + s_down) / 2.
        spl = splrep(t, ts, s=s)
        if len(spl[1]) < n:
            s_up = s
        else:
            s_down = s
        num_iter += 1
        if num_iter > max_iter:
            spl = splrep(t, ts, s=s_down)

    return spl[1][:n]


def get_spline_features(ts, params):
    """
    Function returns extracted features
    Input:
    - ts - time series with three axis
    - params - list of hyperparametes
    Output:
    - features - list with extracted features
    """

    n = params[0]
    x = ts[0]
    y = ts[1]
    z = ts[2]
    m = x.shape[0]
    features = []
    t = np.arange(0, m, 1)
    spl_x = get_spline(t, x, n)
    spl_y = get_spline(t, y, n)
    spl_z = get_spline(t, z, n)
    features = list(np.concatenate((spl_x, spl_y, spl_z), axis=0))

    return features
