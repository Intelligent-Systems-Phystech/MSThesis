from deap import gp, tools, algorithms
import operator
from collections import Counter
import numpy as np
from sklearn.metrics import mean_squared_error
from utils import remove_const


def get_genetic_result(toolbox, pset, primitives, x, y):
    def eval_ind(ind, x, y):
        func = toolbox.compile(ind)
        y_pred = func(x)

        if type(y_pred) == float or type(y_pred) == np.float64:
            y_pred = np.repeat(y_pred, len(y))

        error = mean_squared_error(y, y_pred)
        return error,

    def population(toolbox, n=100):
        i = 0
        pop = []
        while i != n:
            ind = toolbox.individual()
            if check_ind(ind) == 0:
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

    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"),
                                            max_value=17))
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"),
                                              max_value=17))
    toolbox.decorate("mate", gp.staticLimit(key=check_ind, max_value=0))
    toolbox.decorate("mutate", gp.staticLimit(key=check_ind, max_value=0))

    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", np.mean)
    mstats.register("std", np.std)
    mstats.register("min", np.min)
    mstats.register("max", np.max)

    toolbox.register("evaluate", eval_ind, x=x, y=y)
    pop = population(toolbox, n=500)
    for ind in pop:
        if check_ind(ind) == 1:
            print('Error')
    hof = tools.HallOfFame(1)
    pop, log = algorithms.eaSimple(pop, toolbox, 0.5, 0.1, 50, stats=mstats,
                                   halloffame=hof, verbose=False)
    func = gp.compile(hof[0], pset)
    y_pred = func(x)
    return y_pred, hof[0]
