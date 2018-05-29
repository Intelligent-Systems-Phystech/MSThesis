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

# %load_ext autoreload
# %autoreload 2


# %% Good code
toolbox, pset, primitives, variables = prepare_toolbox(params=True)
x_list, y_list, expr_list_param = get_synthetic_data(toolbox, pset,
                                                     primitives, variables,
                                                     n=10000)

print(len(expr_list_param))

expr_list = []
for expr in expr_list_param:
    ind = gp.PrimitiveTree.from_string(expr, pset)
    g = gp.graph(ind)
    expr_list.append(remove_const(primitives, g, 0))


all_idx = list(range(len(expr_list)))
test_idx = list(np.random.choice(np.arange(len(expr_list)), size=500,
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


exprs_greed = pred_to_str(trees_pred, primitives, variables, mat_params,
                          mode='greedy')


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
        y_pred = eval(torch_func)
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()

    y_pred = eval(torch_func).detach().numpy()

    plt.scatter(x_np, y_np, s=5, c='r', label='true')
    plt.plot(x_np, y_pred, label='pred')
    plt.legend()
    plt.savefig('fig/torch_' + str(num))
    plt.close()


# %%
num = 329
x_np = X_test[num, 0:int(X_test.shape[1]/2)]
y_np = X_test[num, int(X_test.shape[1]/2):]
expr_t = expr_list_test_param[num]
expr = exprs_greed[num]
torch_func, w, b = get_torch_func(expr, pset)

x = torch.from_numpy(x_np).type(torch.FloatTensor)
y = torch.from_numpy(y_np).type(torch.FloatTensor)

optimizer = torch.optim.Adam(w + b, lr=0.1)
# optimizer = torch.optim.SGD(w + b, lr=0.1, momentum=0.9)
criterion = nn.MSELoss()
for i in range(1000):
    optimizer.zero_grad()
    y_pred = eval(torch_func)
    loss = criterion(y_pred, y)
    loss.backward()
    optimizer.step()

y_pred = eval(torch_func).detach().numpy()

print(expr_t)
print(torch_func)
print(w)
print(b)

plt.figure(figsize=(8, 6))
plt.scatter(x_np, y_np, c='r', s=10,
            label='$\sin(x - 8.64)\cdot\log(x)$', linewidth=2)
plt.plot(x_np, y_pred,
         label='$(0.53x - 1.15) \cdot (0.44 + 1.46\cos(2.58 + 1.13x))$',
         linewidth=2)
plt.legend(fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel('x', fontsize=14)
plt.ylabel('y', fontsize=14)
plt.savefig('fig/_res_param_4.eps')
plt.show()
