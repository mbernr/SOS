import random as rn
import numpy as np
from utils import *
from deap import algorithms, base, creator, tools

IND_INIT_SIZE = 5
MAX_ITEM = 50
MAX_WEIGHT = 50

items = parse_knapsack("instances/knapsack/low-dimensional/f1_l-d_kp_10_269")
optimal_sol = parse_knapsack_optimal_solution("instances/knapsack/low-dimensional-optimum/f1_l-d_kp_10_269")
NBR_ITEMS = items.number_items


# To assure reproductibility, the RNG seed is set prior to the items
# dict initialization.
rn.seed(64)

creator.create("Fitness", base.Fitness, weights=(1.0,))
creator.create("Individual", set, fitness=creator.Fitness)

toolbox = base.Toolbox()

# Attribute generator
toolbox.register("attr_item", rn.randrange, NBR_ITEMS)

# Structure initializers
toolbox.register("individual", tools.initRepeat, creator.Individual,
    toolbox.attr_item, IND_INIT_SIZE)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def evalKnapsack(individual):
    print(individual)
    weight = 0.0
    value = 0.0
    for item in individual:
        weight += items.weights_items[item]
        value += items.values_items[item]
    if len(individual) > MAX_ITEM or weight > MAX_WEIGHT:
        print("called evalKnapsack on unfeasible solution, returning: 0")
        return 0,             # Ensure overweighted bags are dominated
    print("called evalKnapsack, returning: ", value)
    return value,

def cxSet(ind1, ind2):
    """Apply a crossover operation on input sets. The first child is the
    intersection of the two sets, the second child is the difference of the
    two sets.
    """
    temp = set(ind1)                # Used in order to keep type
    ind1 &= ind2                    # Intersection (inplace)
    ind2 ^= temp                    # Symmetric Difference (inplace)
    return ind1, ind2

def mutSet(individual):
    """Mutation that pops or add an element."""
    if rn.random() < 0.5:
        if len(individual) > 0:     # We cannot pop from an empty set
            individual.remove(rn.choice(sorted(tuple(individual))))
    else:
        print("before: ", individual)
        individual.add(rn.randrange(NBR_ITEMS))
        print("after: ", individual)
    return individual,

toolbox.register("evaluate", evalKnapsack)
toolbox.register("mate", cxSet)
toolbox.register("mutate", mutSet)
toolbox.register("select", tools.selNSGA2)

NGEN = 1000 #number of generations
MU = 100 #numer of individuals to select for next generation
LAMBDA = 200 #number of children to produce on each generation
CXPB = 0.7 #probability next generation is produced by crossover
MUTPB = 0.3 #probability next generation is produced by mutation

pop = toolbox.population(n=MU)

hof = tools.HallOfFame(1)

stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("avg", np.mean, axis=0)
stats.register("std", np.std, axis=0)
stats.register("min", np.min, axis=0)
stats.register("max", np.max, axis=0)

#algorithms.eaMuPlusLambda(pop, toolbox, MU, LAMBDA, CXPB, MUTPB, NGEN, stats,
#                          halloffame=hof)
algorithms.eaSimple(pop, toolbox, 0.7, 0.1, 1000, stats,
                    halloffame=hof)

print("Solution found: ", hof)
print("optimal solution was: ", optimal_sol)

