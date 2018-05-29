import pandas as pd
from deap import gp


def flatten(list_):
    t = []
    for i in list_:
        if not isinstance(i, list):
            t.append(i)
        else:
            t.extend(flatten(i))
    return t


def get_matrix(ind, primitives, variables):
    _, edges, labels = gp.graph(ind)
    tree_mat = pd.DataFrame(0, index=primitives, columns=list(primitives)[1:]
                            + variables)
    if len(edges) == 0:
        tree_mat.loc['*', 'ARG0'] += 1
    else:
        tree_mat.loc['*', labels[edges[0][0]]] += 1
        for edge in edges:
            tree_mat.loc[labels[edge[0]], labels[edge[1]]] += 1
    for var in variables:
        tree_mat[var + '_2'] = tree_mat.apply(lambda row: 1
                                              if row[var] == 2 else 0, axis=1)
        tree_mat[var][tree_mat[var] == 2] = 1
    return tree_mat


def find_parens(s):
    toret = {}
    pstack = []

    for i, c in enumerate(s):
        if c == '(':
            pstack.append(i)
        elif c == ')':
            if len(pstack) == 0:
                raise IndexError("No matching closing parens at: " + str(i))
            toret[pstack.pop()] = i

    if len(pstack) > 0:
        raise IndexError("No matching opening parens at: " + str(pstack.pop()))

    return toret


def remove_const(primitives, graph, vertex):
    nodes, edges, node_dict = graph
    if node_dict[vertex] == 'ARG0':
        return 'ARG0'

    if type(node_dict[vertex]) == float:
        return None

    if node_dict[vertex] in primitives:
        op = node_dict[vertex]
        arity = primitives[op]
        if arity == 1:
            for edge in edges:
                if edge[0] == vertex:
                    res = remove_const(primitives, graph, edge[1])
                    break
            if res is None:
                return None
            else:
                return op + '(' + res + ')'

        if arity == 2:
            res = []
            for edge in edges:
                if edge[0] == vertex:
                    res_tmp = remove_const(primitives, graph, edge[1])
                    res.append(res_tmp)

            if res[0] is not None and res[1] is not None:
                return op + '(' + res[0] + ',' + res[1] + ')'
            if res[0] is None and res[1] is not None:
                return res[1]
            if res[0] is not None and res[1] is None:
                return res[0]
            if res[0] is None and res[1] is None:
                return None
