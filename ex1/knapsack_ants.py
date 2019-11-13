import random
import numpy as np
from deap import algorithms, base, creator, tools
from utils import *
import acopyKS

#ANTS ALGORITHM PARAMETERS
#influence parameters: define wether the pheromones or the edge distance is more important for choosing an edge for an ant
pheromone_influence = 2
distance_influence = 1
rho = 0.03 				# percentage of pheromones that evaporate after each iteration
q = 1 					# amount of pheromone each ant can deposit
ants_count = 50			# if set to None => one ant per node
number_iter = 100

# Parsing TSP instance.
G = parse_knapsack("instances/knapsack/large_scale/knapPI_3_500_1000_1")
optimal_value = parse_knapsack_optimal_solution("instances/knapsack/large_scale-optimum/knapPI_3_500_1000_1")

# Creating Acopy Solver
solver = acopyKS.Solver(rho=rho, q=q)
colony = acopyKS.Colony(alpha=distance_influence, beta=pheromone_influence)
stats_plugin = StatsIterations(lambda sol: sol.value)
solver.add_plugin(stats_plugin)

# Perform algorithm
solution = solver.solve(G, colony, gen_size=ants_count, limit=number_iter)

# retrieve solution
best_solution = solution.value

# Print the best solution we found.
print()
print("Found: {}".format(best_solution))

# Print the optimal solution.
print()
print("Optimal: {}".format(optimal_value))

