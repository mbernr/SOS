import random
import numpy as np
import time
from deap import algorithms, base, creator, tools
from utils import *


# Parsing TSP instance.
G = parse_tsp("instances/tsp/pr76.tsp")
optimal_tour = parse_tsp_optimal_solution("instances/tsp/pr76.opt.tour")

# The evaluation function is the only part that needs to be provided.
# Everything else can be handled by DEAP. 
# This function must return a tuple, since single-solution problems 
# are just a special case of multi-solution problems.
def evaluate(individual):
	return tour_length(G, individual),

# Size of every individual in the population.
# Since every individual represents a hamiltonian cycle, 
# every vertex in G must be part of the individual.
ind_size = G.number_of_nodes()

# We set the weights to (-1.0), since we have a minimization problem
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)


for i in range(100,1001,100):

	# Parameters
	population_size = i
	num_generations = 1000
	prb_crossover = 0.75
	prb_mutation = 0.15
	prb_mutation_shuffle = 0.05
	tournament_size = 50
	verbose = False

	# Individuals are created as random permutations
	toolbox = base.Toolbox()
	toolbox.register("indices", random.sample, range(ind_size), ind_size)
	toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
	toolbox.register("population", tools.initRepeat, list, toolbox.individual)

	# Specify how individuals will be crossed over, mutated, and selected.
	toolbox.register("mate", tools.cxPartialyMatched)
	toolbox.register("mutate", tools.mutShuffleIndexes, indpb=prb_mutation_shuffle)
	toolbox.register("select", tools.selTournament, tournsize=tournament_size)
	toolbox.register("evaluate", evaluate)

	# Initialize population.
	#random.seed(169)
	population = toolbox.population(n=population_size)

	# Keep the (single) best solution. 
	best_solutions = tools.HallOfFame(1)

	# Record statistics.
	stats = tools.Statistics(lambda ind: ind.fitness.values)
	stats.register("avg", np.mean)
	stats.register("std", np.std)
	stats.register("min", np.min)
	stats.register("max", np.max)

	start_time = time.time()

	# Execute algorithm provided by DEAP.
	algorithms.eaSimple(
		population, 
		toolbox, 
		prb_crossover, 
		prb_mutation, 
		num_generations, 
		stats=stats, 
		halloffame=best_solutions,
		verbose=verbose
	)

	end_time = time.time()
	run_time = round(end_time - start_time, 2)

	# Format the best solution to always start from vertex 0
	best_solution = start_tour_from_zero(best_solutions[0])

	if verbose:
		# Print the best solution we found.
		print()
		print("Found: {} ({}s)".format(evaluate(best_solution)[0], run_time))
		print(best_solution)

		# Print the optimal solution.
		print()
		print("Optimal: {}".format(evaluate(optimal_tour)[0]))
		print(optimal_tour)
	else:
		print("{}\t{}\t{}\t{}\t{}\t{}".format(
			population_size, 
			prb_crossover, 
			prb_mutation, 
			evaluate(best_solution)[0],
			evaluate(optimal_tour)[0],
			run_time
		))


