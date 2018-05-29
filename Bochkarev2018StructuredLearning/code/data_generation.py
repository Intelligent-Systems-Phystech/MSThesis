import random
import string
from collections import Counter

import numpy as np
import pandas as pd
from deap import base, creator, gp, tools

from genetic import get_genetic_result
from utils import get_matrix, remove_const


def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))


def inv(a):
    return np.true_divide(1, a, where=(a != 0))


def log(a):
    return np.log(np.array(a).clip(min=0.01))


def exp(a):
    return np.exp(np.array(a).clip(max=10))


def sqrt(a):
    return np.sqrt(np.array(a).clip(min=0))


def square(a):
    return a**2


def prepare_toolbox(params=False, n_var=1):
    """
    Prepare deap toolbox.

    Returns:
        toolbox: Toolbox object with registered fuctions.
        pset: Primitive set
        primitives: List of primitives used for model generation.
    """

    if 'Individual' not in dir(creator):
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", gp.PrimitiveTree,
                       fitness=creator.FitnessMin)

    primitives = {'*': 1}

    pset = gp.PrimitiveSet("MAIN", n_var)

    pset.addPrimitive(np.add, 2)
    primitives['add'] = 2

    pset.addPrimitive(np.multiply, 2)
    primitives['multiply'] = 2

    pset.addPrimitive(np.sin, 1)
    primitives['sin'] = 1

    pset.addPrimitive(np.cos, 1)
    primitives['cos'] = 1

    pset.addPrimitive(exp, 1)
    primitives['exp'] = 1

    pset.addPrimitive(log, 1)
    primitives['log'] = 1

    pset.addPrimitive(sqrt, 1)
    primitives['sqrt'] = 1

    pset.addPrimitive(square, 1)
    primitives['square'] = 1

    if params is True:
        pset.addEphemeralConstant(id_generator(),
                                  lambda: random.uniform(-10, 10))

    variables = list(pset.arguments)

    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genGrow, pset=pset, min_=1, max_=8)
    toolbox.register("individual", tools.initIterate, creator.Individual,
                     toolbox.expr)
    toolbox.register("compile", gp.compile, pset=pset)

    return toolbox, pset, primitives, variables


def get_synthetic_data(toolbox, pset, primitives, variables, n=100000):
    def population(toolbox, n=100):
        i = 0
        pop = []
        while i != n:
            ind = toolbox.individual()
            vertices = list(gp.graph(ind)[2].values())
            if check_ind(ind) == 0 and 'ARG0' in vertices:
                pop.append(ind)
                i += 1
        return pop

    def check_ind(ind):
        new_ind = remove_const(primitives, gp.graph(ind), 0)
        if new_ind is None:
            return 1
        ind = gp.PrimitiveTree.from_string(new_ind, pset)
        cnt = Counter()
        for k, v in gp.graph(ind)[2].items():
            if v in primitives:
                cnt[v] += 1

        if len(cnt) == 0:
            return 0

        max_repeat = cnt.most_common(1)[0][1]

        if max_repeat > 1:
            return 1
        else:
            return 0
    toolbox.register("expr", gp.genGrow, pset=pset, min_=3, max_=8)
    x_list = []
    y_list = []

    pop = population(toolbox, n=n)
    expr_list = []
    for i in pop:
        expr_list.append(str(i))

    expr_list = list(set(expr_list))

    for expr in expr_list:
        X = np.random.uniform(-5, 5, 100)
        X = np.sort(X)

        ind = gp.PrimitiveTree.from_string(expr, pset)
        func = gp.compile(ind, pset)
        y = func(X) + 0.02*np.random.randn(len(X))

        x_list.append(X)
        y_list.append(y)
    return x_list, y_list, expr_list


# def tree_generator(matrix, op, primitives, variables):
#     if op in variables:
#         yield (op, matrix)
#         return
#
#     arity = primitives[op]
#     if arity == 1:
#         for next_op in matrix.columns:
#             if matrix.loc[op, next_op] < 0:
#                 continue
#
#             matrix_next = matrix.copy()
#             if next_op not in variables:
#                 matrix_next.loc[:, next_op] = -1
#             for expr, mat in tree_generator(matrix_next, next_op,
#                                             primitives, variables):
#                 yield ([op, '(', expr, ')'], mat)
#
#     if arity == 2:
#         candidate_list = (
#             list(itertools.combinations(matrix.columns, 2))
#             + list(itertools.combinations_with_replacement(variables, 2)))
#         for left_op, right_op in candidate_list:
#             if matrix.loc[op, left_op] < 0:
#                 continue
#             if matrix.loc[op, right_op] < 0:
#                 continue
#
#             matrix_n = matrix.copy()
#             if left_op not in variables:
#                 matrix_n.loc[:, left_op] = -1
#             if right_op not in variables:
#                 matrix_n.loc[:, right_op] = -1
#
#             for expr_l, mat_l in tree_generator(matrix_n, left_op,
#                                                 primitives, variables):
#                 for expr_r, mat_r in tree_generator(mat_l, right_op,
#                                                     primitives, variables):
#                     yield ([op, '(', expr_l, ',', expr_r, ')'], mat_r)


def get_model_data(x_list, y_list, expr_list, pset, primitives, variables):
    X = np.empty((len(x_list), len(x_list[0])+len(y_list[0])))
    for i, (x, y) in enumerate(zip(x_list, y_list)):
        X[i, :] = np.concatenate((x, y))

    mat = get_matrix(gp.PrimitiveTree.from_string(expr_list[0], pset),
                     primitives, variables)
    shape = mat.shape
    columns = list(mat.columns.values)
    index = list(mat.index.values)

    trees = np.empty((len(expr_list), shape[0]*shape[1]))
    for i, expr in enumerate(expr_list):
        ind = gp.PrimitiveTree.from_string(expr, pset)
        trees[i, :] = get_matrix(ind, primitives, variables).values.flatten()
    return X, trees, (shape, columns, index)


def get_real_data(toolbox, pset, primitives, mode, n=10):
    def check_ind(ind):
        new_ind = remove_const(primitives, gp.graph(ind), 0)
        if new_ind is None:
            return 1
        ind = gp.PrimitiveTree.from_string(new_ind, pset)
        cnt = Counter()
        for k, v in gp.graph(ind)[2].items():
            if v in primitives:
                cnt[v] += 1

        if len(cnt) == 0:
            return 0

        max_repeat = cnt.most_common(1)[0][1]

        if max_repeat > 1:
            return 1
        else:
            return 0

    x_list = []
    y_list = []
    expr_list = []

    if mode == 'exchange':
        data = pd.read_csv('../data/daily-foreign-exchange-rates-31-.csv',
                           names=['x', 'y'], header=0,
                           dtype={'x': str, 'y': np.float}, nrows=4770)
    if mode == 'stock':
        data = pd.read_csv('../data/ibm-common-stock-closing-prices.csv',
                           names=['x', 'y'], header=0,
                           dtype={'x': str, 'y': np.float}, nrows=1007)
    data = data.iloc[:-1]
    nrows = len(data)
    num_points = 100
    n_done = 0
    while n_done < n:
        start = np.random.randint(0, nrows - num_points)
        x = data.iloc[start:start+num_points, 0].values
        x = np.linspace(-3, 3, num=num_points)
        y = data.iloc[start:start+num_points, 1].values
        if mode == 'stock':
            y = np.log(y)
        y_gen, ind = get_genetic_result(toolbox, pset, primitives, x, y)
        if check_ind(ind) == 0:
            x_list.append(list(x))
            y_list.append(list(y))
            expr_list.append(str(ind))
            n_done += 1
    return x_list, y_list, expr_list
