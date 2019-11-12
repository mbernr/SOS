import random
import numpy as np
from deap import algorithms, base, creator, tools
from utils import *


# Parsing TSP instance.
G = parse_tsp("instances/tsp/bays29.tsp")
optimal_tour = parse_tsp_optimal_solution("instances/tsp/bays29.opt.tour")

# The evaluation function is the only part that needs to be provided.
# Everything else can be handled by DEAP. 
# This function must return a tuple, since single-solution problems 
# are just a special case of multi-solution problems.
def evaluate(individual):
	return tour_length(G, individual),


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

# Format the best solution to always start from vertex 0
best_solution = start_tour_from_zero(best_solutions[0])

# Print the best solution we found.
print()
print("Found: {}".format(evaluate(best_solution)[0]))
print(best_solution)

# Print the optimal solution.
print()
print("Optimal: {}".format(evaluate(optimal_tour)[0]))
print(optimal_tour)

