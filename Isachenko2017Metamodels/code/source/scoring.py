import numpy as np
import pandas as pd
import scipy as sc
import time

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

def get_internal_score(clf, X, y, max_iter=11):
    """
    Function implement validation procedure
    Input:
    - clf - classifier model
    - X - dataset that contains extracted features
    - y - vector of activity labels
    - max_iter - number of iteration of splitting data into train/test
    Output:
    - scores - list with multiclass score and binary scores
    """

    nb = np.unique(y).shape[0]
    scores = np.zeros(nb+1)
    for j in range(max_iter):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
        clf.fit(X_train, y_train)
        y_predict = clf.predict(X_test)
        scores[0] += accuracy_score(y_test, y_predict)
        for i in range(nb):
            scores[i+1] += accuracy_score(1*(np.array(y_test) == i), 
                                          1*(np.array(y_predict) == i))

    return scores / max_iter


def get_score(df, estimator, params_grid, test_size=0.3):
    """
    Function makes something strange
    """

    X = df.loc[:, df.columns != 'activity'].values
    y = df['activity'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    clf = GridSearchCV(estimator, params_grid)
    clf.fit(X_train, list(y_train))
    clf_lr = clf.best_estimator_
    scores = get_internal_score(clf_lr, X, list(y))

    return scores
