
import matplotlib.pyplot as plt
import numpy as np
from deap import gp
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from data_generation import get_model_data, get_synthetic_data, prepare_toolbox
from model import model, pred_to_str

# %load_ext autoreload
# %autoreload 2


# %% Good code
toolbox, pset, primitives, variables = prepare_toolbox()
x_list, y_list, expr_list = get_synthetic_data(toolbox, pset,
                                               primitives, variables)

print(len(expr_list))

X, trees, mat_params = get_model_data(
    x_list, y_list, expr_list, pset, primitives, variables)

X_train, X_test, trees_train, trees_test = train_test_split(
    X, trees, test_size=100)

trees_pred = model(X_train, trees_train, X_test, trees_test, mode='knn')
exprs_greed = pred_to_str(trees_pred, primitives, variables, mat_params,
                          mode='greedy')
exprs_mean = pred_to_str(trees_pred, primitives, variables, mat_params,
                         mode='mean')
exprs_prob = pred_to_str(trees_pred, primitives, variables, mat_params,
                         mode='probability')
exprs_true = pred_to_str(trees_test, primitives, variables, mat_params,
                         mode='greedy')

# %% Demo
mse_mean = []
mse_greed = []
mse_prob = []
for num in range(len(exprs_true)):
    expr_m = exprs_mean[num]
    expr_g = exprs_greed[num]
    expr_p = exprs_prob[num]
    expr_t = exprs_true[num]

    X_test[num]
    x = X_test[num, 0:int(X_test.shape[1]/2)]
    y = X_test[num, int(X_test.shape[1]/2):]

    tree_m = gp.PrimitiveTree.from_string(expr_m, pset)
    func = gp.compile(tree_m, pset)
    y_m = func(x)
    mse_mean.append(mean_squared_error(y, y_m))

    tree_g = gp.PrimitiveTree.from_string(expr_g, pset)
    func = gp.compile(tree_g, pset)
    y_g = func(x)
    mse_greed.append(mean_squared_error(y, y_g))

    tree_p = gp.PrimitiveTree.from_string(expr_p, pset)
    func = gp.compile(tree_p, pset)
    y_p = func(x)
    mse_prob.append(mean_squared_error(y, y_p))

    # plt.scatter(x, y, c='r', s=3, label=expr_t + '  TRUE')
    # plt.plot(x, y_m, label=expr_m + ' mean')
    # plt.plot(x, y_g, label=expr_g + ' greedy')
    # plt.plot(x, y_p, label=expr_p + ' prob')
    # plt.legend()
    # plt.savefig('fig/result' + str(num))
    # plt.close()

print(f'Greedy algorithm: {np.median(mse_greed):.4}')
print(f'Dynamic mean: {np.median(mse_mean):.4}')
print(f'Dynamic probability: {np.median(mse_prob):.4}')

# %%
num = 63
expr_m = exprs_mean[num]
expr_g = exprs_greed[num]
expr_p = exprs_prob[num]
expr_t = exprs_true[num]

X_test[num]
x = X_test[num, 0:int(X_test.shape[1]/2)]
y = X_test[num, int(X_test.shape[1]/2):]

tree_m = gp.PrimitiveTree.from_string(expr_m, pset)
func = gp.compile(tree_m, pset)
y_m = func(x)
mse_mean.append(mean_squared_error(y, y_m))

tree_g = gp.PrimitiveTree.from_string(expr_g, pset)
func = gp.compile(tree_g, pset)
y_g = func(x)
mse_greed.append(mean_squared_error(y, y_g))

tree_p = gp.PrimitiveTree.from_string(expr_p, pset)
func = gp.compile(tree_p, pset)
y_p = func(x)
mse_prob.append(mean_squared_error(y, y_p))

print(expr_m)
print(expr_g)
print(expr_p)
print(expr_t)


plt.figure(figsize=(8, 6))
plt.scatter(x, y, c='r', s=10, label='$\log(x\sqrt{e^{\cos(x)}})$')
plt.plot(x, y_m, label='$\log(2x)$   mean', linewidth=2)
plt.plot(x, y_g, label='$\log(\sqrt{x + \sin(x)})$   greedy', linewidth=2)
plt.plot(x, y_p, label='$\log(x)$   prob', linewidth=2)
plt.xlabel('x', fontsize=14)
plt.ylabel('y', fontsize=14)
plt.legend(fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.savefig('fig/_non_param_4.eps')
