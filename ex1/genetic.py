from utils import path_length, parse_tsp

import random
import numpy as np
from deap import algorithms, base, creator, tools


# Parsing TSP instance.
edgelist = "tsp_instances/0010_k1.txt"
coords = "tsp_instances/kroB150_k3_2.txt"
G = parse_tsp(edgelist)

# The evaluation function is the only part that needs to be provided.
# Everything else can be handled by DEAP. 
# This function must return a tuple, since single-solution problems 
# are just a special case of multi-solution problems.
def evaluate(individual):
    return path_length(G, individual), 


# Size of every individual in the population.
# Since every individual represents a hamiltonian cycle, 
# every vertex in G must be part of the individual.
IND_SIZE = G.number_of_nodes()

# We set the weights to (-1.0), since we have a minimization problem
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

# Individuals are created as random permutations
toolbox = base.Toolbox()
toolbox.register("indices", random.sample, range(IND_SIZE), IND_SIZE)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Specify how individuals will be crossed over, mutated, and selected.
toolbox.register("mate", tools.cxPartialyMatched)
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", evaluate)

# Initialize population.
random.seed(169)
pop = toolbox.population(n=300)

# Keep the (single) best solution. 
best_solutions = tools.HallOfFame(1)

# Record statistics.
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("avg", np.mean)
stats.register("std", np.std)
stats.register("min", np.min)
stats.register("max", np.max)

# Execute algorithm provided by DEAP.
algorithms.eaSimple(pop, toolbox, 0.7, 0.2, 40, stats=stats, 
                    halloffame=best_solutions)

# Print the best solution.
print(best_solutions[0], evaluate(best_solutions[0]))