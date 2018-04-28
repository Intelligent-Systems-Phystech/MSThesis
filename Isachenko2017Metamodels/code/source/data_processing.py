import numpy as np
import pandas as pd
import scipy as sc
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from scipy.interpolate import splrep, splev, splprep


def check_candidate(candidate, data_type, threshold=2.*1e8):
    """
    Function checks the validity of time series
    """

    if data_type == "USCHAD":
        threshold = 0.
    tsp = np.array(candidate['timestamp'])
    diffs = tsp[1:] - tsp[:-1]

    return np.sum(diffs > threshold) == 0


def get_time_series(accelerations, data_type, nb=200):
    """
    Function extracts time series along three axis from raw data
    Input:
    - accelerations - raw data from one user with specific activity
    - data_type - string: {'WISDM', 'USCHAD'}
    Output:
    - TS - list whose elements time series
    """

    accelerations.index = [i for i in range(len(accelerations))]
    TS = []
    st = 0
    fi = st + nb
    while fi < len(accelerations):
        candidate = accelerations.loc[[st + i for i in range(nb)], :]
        if check_candidate(candidate, data_type):
            TS.append([np.array(candidate['x']),
                       np.array(candidate['y']),
                       np.array(candidate['z'])])
        st = fi
        fi += nb

    return TS


def get_feature_matrix(data, data_type, get_feature_names,
                       get_features, params=[]):
    """
    Function creates dataframe with extracted features
    Input:
    - data - dataset which contains columns:
      {'user_id', 'activity', 'timestamp', 'x', 'y', 'z'}
    - data_type - string: {'WISDM', 'USCHAD'}
    - get_feature_names - function that returns list of
      columns names for the dataframe
    - get_features - function that returns extracted feature for each ts
    - params - internal parameters for feature extractyion procedure
    Output:
    - df - dataframe which rows represent time series and columns
      are extracted features
    """

    classes = list(set(data['activity']))
    feature_names = get_feature_names(params)
    df = pd.DataFrame(columns=['activity']+feature_names)

    id_range = np.unique(np.array(data['id_user']))
    for id_user in id_range:
        for activity in classes:
            mask = (data.loc[:, 'id_user'] == id_user) & \
                        (data.loc[:, 'activity'] == activity)
            accelerations = data.loc[mask, ['timestamp', 'x', 'y', 'z']]
            TS = get_time_series(accelerations, data_type, nb=200)
            for ts in TS:
                features = get_features(ts, params)
                df.loc[len(df), :] = [classes.index(activity)] + features
    return df


def get_distribution(data, df):
    """
    Function returns distribution of time series by activities
    Input:
    - data - dataset which contains columns:
      {'user_id', 'activity', 'timestamp', 'x', 'y', 'z'}
    - df - dataframe which rows represent time series and columns
      are extracted features
    Output:
    - printed info
    """

    classes = list(set(data['activity']))
    for activity in classes:
        nb = np.sum(df['activity'] == classes.index(activity))
        print("{:<20}{:<9d}{:<5.2f} %".format(activity, nb,
                                              100. * nb / df.shape[0]))
    print("")
    print("Number of objects: {:d}".format(df.shape[0]))
