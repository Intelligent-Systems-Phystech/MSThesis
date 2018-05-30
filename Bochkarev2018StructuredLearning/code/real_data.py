# %%
import matplotlib.pyplot as plt
import numpy as np
import torch
from deap import gp
from torch import nn

from data_generation import get_model_data, get_synthetic_data, prepare_toolbox
from model import model, pred_to_str
from parameters import get_torch_func
from utils import remove_const

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
import numpy as np
from time import time
from data_generation import get_real_data
from collections import Counter

from sklearn.metrics import mean_squared_error

from genetic import get_genetic_result

# %load_ext autoreload
# %autoreload 2

# %%
toolbox, pset, primitives, variables = prepare_toolbox(params=True)

start = time()
x_list, y_list, expr_list_param = get_real_data(toolbox, pset, primitives,
                                                mode='stock', n=100)
print(time() - start)
print('ok!')
print(len(expr_list_param))

expr_list = []
for expr in expr_list_param:
    ind = gp.PrimitiveTree.from_string(expr, pset)
    g = gp.graph(ind)
    expr_list.append(remove_const(primitives, g, 0))

all_idx = list(range(len(expr_list)))
test_idx = list(np.random.choice(np.arange(len(expr_list)), size=10,
                                 replace=False))
train_idx = [i for i in all_idx if i not in test_idx]


x_list_train = [x_list[i] for i in train_idx]
y_list_train = [y_list[i] for i in train_idx]
expr_list_train = [expr_list[i] for i in train_idx]
expr_list_train_param = [expr_list_param[i] for i in train_idx]

x_list_test = [x_list[i] for i in test_idx]
y_list_test = [y_list[i] for i in test_idx]
expr_list_test = [expr_list[i] for i in test_idx]
expr_list_test_param = [expr_list_param[i] for i in test_idx]

X_train, trees_train, mat_params = get_model_data(
    x_list_train, y_list_train, expr_list_train, pset, primitives, variables)
X_test, trees_test, mat_params = get_model_data(
    x_list_test, y_list_test, expr_list_test, pset, primitives, variables)

trees_pred = model(X_train, trees_train, X_test, trees_test)
start = time()
exprs_greed = pred_to_str(trees_pred, primitives, variables, mat_params,
                          mode='greedy')

mse_gen = []
mse_torch = []
for num, expr in enumerate(exprs_greed):
    x_np = X_test[num, 0:int(X_test.shape[1]/2)]
    y_np = X_test[num, int(X_test.shape[1]/2):]
    expr_t = expr_list_test_param[num]
    expr = exprs_greed[num]
    torch_func, w, b = get_torch_func(expr, pset)

    x = torch.from_numpy(x_np).type(torch.FloatTensor)
    y = torch.from_numpy(y_np).type(torch.FloatTensor)

    optimizer = torch.optim.Adam(w + b, lr=0.1)
    criterion = nn.MSELoss()
    for i in range(1000):
        optimizer.zero_grad()
        try:
            y_pred = eval(torch_func)
        except AttributeError:
            continue
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()

    try:
        y_pred = eval(torch_func).detach().numpy()
        y_gen = gp.compile(expr_t, pset)(x_np)
        mse_gen.append(mean_squared_error(y, y_gen))
        mse_torch.append(mean_squared_error(y, y_pred))
    except AttributeError:
        continue
print(time() - start)
print(np.mean(mse_gen))
print(np.mean(mse_torch))
