import torch
from deap import gp

from utils import find_parens


def get_torch_func(expr, pset):
    replace_dict = {
        'ARG0': ('x', ''),
        'cos': ('torch.cos', ''),
        'sin': ('torch.sin', ''),
        'add': ('torch.add', ''),
        'multiply': ('torch.mul', ''),
        'exp': ('torch.exp(torch.clamp', ', max=10)'),
        'log': ('torch.log(torch.clamp', ', min=0.01)'),
        'square': ('', '**2'),
        'sqrt': ('torch.sqrt(torch.clamp', ', min=0.01)'),
    }

    ind = gp.PrimitiveTree.from_string(expr, pset)
    graph = gp.graph(ind)
    binary = ['add', 'multiply']
    replace_list = [v for (k, v) in graph[2].items()]

    w = []
    b = []
    num_param = 0
    torch_func = expr
    for op in replace_list:
        if op in binary:
            torch_func = torch_func.replace(op, replace_dict[op][0], 1)
            continue

        w_i = torch.ones(1, requires_grad=True)
        b_i = torch.zeros(1, requires_grad=True)

        w.append(w_i)
        b.append(b_i)

        if op != 'ARG0':
            pars = find_parens(torch_func)
            end_bracket = pars[torch_func.find(op) + len(op)]
            torch_func = (torch_func[:end_bracket] + replace_dict[op][1]
                          + torch_func[end_bracket:])
        torch_func = torch_func.replace(op, f'b[{num_param}] + '
                                        + f'w[{num_param}]*'
                                        + replace_dict[op][0], 1)
        num_param += 1
    return torch_func, w, b
