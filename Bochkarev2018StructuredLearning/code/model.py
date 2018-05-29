import itertools

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier

from utils import flatten


def model(X_train, trees_train, X_test, trees_test, mode='forest'):
    if mode == 'forest':
        clf = RandomForestClassifier(n_estimators=500, n_jobs=-1)
    if mode == 'knn':
        clf = KNeighborsClassifier()
    if mode == 'extra':
        clf = ExtraTreesClassifier()
    clf.fit(X_train, trees_train)
    trees_pred_raw = clf.predict_proba(X_test)
    trees_pred = np.empty(trees_test.shape)
    for i, output in enumerate(trees_pred_raw):
        if output.shape[1] == 1:
            trees_pred[:, i] = 0
        else:
            trees_pred[:, i] = output[:, 1]
    return trees_pred


def get_prob_matrices(trees_pred, mat_params):
    shape, columns, index = mat_params
    mats = []
    for i in range(trees_pred.shape[0]):
        mat = pd.DataFrame(trees_pred[i].reshape(shape),
                           index=index, columns=columns)
        mats.append(mat)
    return mats


def mat_to_string(mat, op, primitives, variables):
    mat = mat.copy()
    if op in variables:
        return op.replace('_2', '')

    arity = primitives[op]

    if arity == 1:
        next_op = mat.loc[op].idxmax(axis=1)
        mat.loc[op, next_op] = -1
        return [op, '(', mat_to_string(mat, next_op, primitives, variables),
                ')']

    if arity == 2:
        next_op1 = mat.loc[op].idxmax(axis=1)
        mat.loc[op, next_op1] = -1
        next_op2 = mat.loc[op].idxmax(axis=1)
        mat.loc[op, next_op2] = -1
        return [op, '(', mat_to_string(mat, next_op1, primitives, variables),
                ',', mat_to_string(mat, next_op2, primitives, variables), ')']


def mat_to_string_dynamic(mat, op, primitives, variables, done_dict):
    matrix_local = mat.copy()
    ops_used = frozenset([op for op in mat.columns
                          if mat.loc['*', op] == -1])

    if ops_used in done_dict:
        return done_dict[ops_used]

    if op in variables:
        result = (mat, op.replace('_2', ''), 1)
        done_dict[ops_used] = result
        return result

    arity = primitives[op]
    if arity == 1:
        matrices = []
        op_candidates = []
        subtree_likelihood = []

        for next_op in matrix_local.columns:
            if matrix_local.loc[op, next_op] <= 0:
                continue

            matrix_next = matrix_local.copy()
            matrix_next.loc[:, next_op] = -1
            matrix_next, op_candidate, likelihood = mat_to_string_dynamic(
                matrix_next, next_op, primitives, variables, done_dict)
            op_candidates.append(op_candidate)
            subtree_likelihood.append(likelihood
                                      * matrix_local.loc[op, next_op])
            matrices.append(matrix_next)

        if len(matrices) == 0:
            return (mat, None, 0)

        probs = np.array(subtree_likelihood)
        next_ind = np.argmax(probs)

        matrix_next = matrices[next_ind]
        next_op = op_candidates[next_ind]
        probability = probs[next_ind]
        result = (matrix_next, [op, '(', next_op, ')'], probability)
        done_dict[ops_used] = result
        return result

    if arity == 2:
        matrices = []
        op_candidates = []
        subtree_likelihood = []

        for left_op, right_op in itertools.permutations(
                matrix_local.columns, 2):
            if matrix_local.loc[op, left_op] <= 0:
                continue
            if matrix_local.loc[op, right_op] <= 0:
                continue

            matrix_next = matrix_local.copy()
            matrix_next.loc[:, left_op] = -1
            matrix_next.loc[:, right_op] = -1
            matrix_left, op_candidate_l, likelihood_l = mat_to_string_dynamic(
                matrix_next, left_op, primitives, variables, done_dict)
            matrix_right, op_candidate_r, likelihood_r = mat_to_string_dynamic(
                matrix_left, right_op, primitives, variables, done_dict)

            op_candidates.append((op_candidate_l, op_candidate_r))
            subtree_likelihood.append(likelihood_l
                                      * matrix_local.loc[op][left_op]
                                      * likelihood_r
                                      * matrix_local.loc[op][right_op])
            matrices.append(matrix_right)

        if len(matrices) == 0:
            return (mat, None, 0)
        probs = np.array(subtree_likelihood)

        next_ind = np.argmax(probs)
        matrix_next = matrices[next_ind]
        left_op, right_op = op_candidates[next_ind]
        probability = probs[next_ind]
        result = (matrix_next, [op, '(', left_op, ',', right_op, ')'],
                  probability)
        done_dict[ops_used] = result
        return result


def mat_to_string_mean(mat, op, primitives, variables, done_dict):
    matrix_local = mat.copy()
    ops_used = frozenset([op for op in mat.columns
                          if mat.loc['*', op] == -1])

    if ops_used in done_dict:
        return done_dict[ops_used]

    if op in variables:
        result = (mat, op.replace('_2', ''), 0, 0)
        done_dict[ops_used] = result
        return result

    arity = primitives[op]
    if arity == 1:
        matrices = []
        op_candidates = []
        subtree_mean = []
        subtree_length = []

        for next_op in matrix_local.columns:
            if matrix_local.loc[op, next_op] <= 0:
                continue

            matrix_next = matrix_local.copy()
            matrix_next.loc[:, next_op] = -1
            matrix_next, op_candidate, mean, length = mat_to_string_mean(
                matrix_next, next_op, primitives, variables, done_dict)

            op_candidates.append(op_candidate)
            subtree_mean.append((mean*length + matrix_local.loc[op, next_op])
                                / (length + 1))
            subtree_length.append(length + 1)

            matrices.append(matrix_next)

        if len(matrices) == 0:
            result = (mat, None, 0, 1e5)
            done_dict[ops_used] = result
            return result

        subtree_mean = np.array(subtree_mean)
        next_ind = np.argmax(subtree_mean)

        matrix_next = matrices[next_ind]
        next_op = op_candidates[next_ind]
        mean = subtree_mean[next_ind]
        length = subtree_length[next_ind]

        result = (matrix_next, [op, '(', next_op, ')'], mean, length)
        done_dict[ops_used] = result
        return result

    if arity == 2:
        matrices = []
        op_candidates = []
        subtree_mean = []
        subtree_length = []

        for left_op, right_op in itertools.permutations(
                matrix_local.columns, 2):
            if matrix_local.loc[op, left_op] <= 0:
                continue
            if matrix_local.loc[op, right_op] <= 0:
                continue

            matrix_next = matrix_local.copy()
            matrix_next.loc[:, left_op] = -1
            matrix_next.loc[:, right_op] = -1
            res = mat_to_string_mean(matrix_next, left_op,
                                     primitives, variables, done_dict)
            matrix_left, op_candidate_l, mean_l, length_l = res
            res = mat_to_string_mean(matrix_left, right_op,
                                     primitives, variables, done_dict)
            matrix_right, op_candidate_r, mean_r, length_r = res

            op_candidates.append((op_candidate_l, op_candidate_r))
            subtree_mean.append((mean_l*length_l + mean_r*length_r
                                 + matrix_local.loc[op, left_op]
                                 + matrix_local.loc[op, right_op])
                                / (length_l + length_r + 2))
            subtree_length.append(length_l + length_r + 2)
            matrices.append(matrix_right)

        if len(matrices) == 0:
            result = (mat, None, 0, 1e5)
            done_dict[ops_used] = result
            return result

        subtree_mean = np.array(subtree_mean)

        next_ind = np.argmax(subtree_mean)
        matrix_next = matrices[next_ind]
        left_op, right_op = op_candidates[next_ind]
        mean = subtree_mean[next_ind]
        length = subtree_length[next_ind]

        result = (matrix_next, [op, '(', left_op, ',', right_op, ')'],
                  mean, length)
        done_dict[ops_used] = result
        return result


def pred_to_str(trees_pred, primitives, variables, mat_params, mode):

    strings = []
    matrices = get_prob_matrices(trees_pred, mat_params)

    variables = flatten([[var, var+'_2'] for var in variables])

    if mode == 'greedy':
        for i, matrix in enumerate(matrices):
            mat = matrix.copy()
            op = '*'
            res = mat_to_string(mat, op, primitives, variables)
            res_flat = flatten(res)
            res_flat = res_flat[2:-1]
            strings.append(''.join(res_flat))

    if mode == 'probability':
        for i, matrix in enumerate(matrices):
            mat = matrix.copy()
            op = '*'
            _, res, _ = mat_to_string_dynamic(mat, op, primitives,
                                              variables, {})
            res_flat = flatten(res)
            res_flat = res_flat[2:-1]
            strings.append(''.join(res_flat))

    if mode == 'mean':
        for i, matrix in enumerate(matrices):
            mat = matrix.copy()
            op = '*'
            _, res, mean, _ = mat_to_string_mean(mat, op, primitives,
                                                 variables, {})
            res_flat = flatten(res)
            if len(res_flat) == 3:
                strings.append(''.join(res_flat[2]))
                continue
            res_flat = res_flat[2:-1]
            strings.append(''.join(res_flat))

    return strings
