import random
import numpy as np
from deap import algorithms, base, creator, tools
from utils import *
import acopyKS

#ANTS ALGORITHM PARAMETERS
#influence parameters: define wether the pheromones or the edge distance is more important for choosing an edge for an ant
pheromone_influence = 3
distance_influence = 1 
rho = 0.03 				# percentage of pheromones that evaporate after each iteration
q = 3 					# amount of pheromone each ant can deposit
ants_count = 50			# if set to None => one ant per node
number_iter = 1000

# Parsing TSP instance.
G = parse_knapsack("instances/knapsack/low-dimensional/f1_l-d_kp_10_269")
optimal_tour = parse_tsp_optimal_solution("instances/tsp/pr76.opt.tour")

# Creating Acopy Solver
solver = acopy.Solver(rho=rho, q=q)
colony = acopy.Colony(alpha=distance_influence, beta=pheromone_influence)
stats_plugin = StatsIterations(lambda sol: sol.cost)
solver.add_plugin(stats_plugin)

# Perform algorithm
tour = solver.solve(G, colony, gen_size=ants_count, limit=number_iter)

# retrieve solution
best_solution = tour.nodes

# Print the best solution we found.
print()
print("Found: {}".format(tour_length(G, best_solution)))
print(best_solution)

# Print the optimal solution.
print()
print("Optimal: {}".format(tour_length(G, optimal_tour)))
print(optimal_tour)

