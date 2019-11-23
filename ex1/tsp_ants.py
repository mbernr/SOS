import random
import numpy as np
from deap import algorithms, base, creator, tools
from utils import *
import acopy
import time

#random.seed(0)

#ANTS ALGORITHM PARAMETERS
#influence parameters: define wether the pheromones or the edge distance is more important for choosing an edge for an ant
pheromone_influence = 3
distance_influence = 1
rho = 0.00 				# percentage of pheromones that evaporate after each iteration
q = 1 					# amount of pheromone each ant can deposit
ants_count = 200			# if set to None => one ant per node
print("{},{},{},{},{},{},{}".format("beta","alpha","nAnts","evap","time","best_sol","score"))
for ab in [(3,1)]:
	for ants_count in [20]:
		for pheromone_evap in [0.04]:
			number_iter = 1000

			start = time.time()
						
			pheromone_influence = ab[0]
			distance_influence = ab[1]
			rho = pheromone_evap				# percentage of pheromones that evaporate after each iteration
			q = 1 					# amount of pheromone each ant can deposit
			ants_count = ants_count			# if set to None => one ant per node
			number_iter = 100

			# Parsing TSP instance.
			G = parse_tsp("instances/tsp/ch130.tsp")
			optimal_tour = parse_tsp_optimal_solution("instances/tsp/ch130.opt.tour")

			# Creating Acopy Solver
			solver = acopy.Solver(rho=rho, q=q)
			colony = acopy.Colony(alpha=distance_influence, beta=pheromone_influence)
			listBool = [False]
			time_limit = acopy.plugins.TimeLimit(seconds=30)
			#stats_plugin = StatsIterations(lambda sol: sol.cost)
			#solver.add_plugin(stats_plugin)
			solver.add_plugin(time_limit)

			# Perform algorithm
			tour = solver.solve(G, colony, gen_size=ants_count, limit=number_iter)

			# retrieve solution
			best_solution = tour.nodes

			best_sol_val = tour_length(G, best_solution)
			optimal_sol_val = tour_length(G, optimal_tour)


			end = time.time() - start

			
			print("{},{},{},{},{},{},{}".format(ab[0], ab[1], ants_count, pheromone_evap, end, best_sol_val, best_sol_val/optimal_sol_val))

			# Print the best solution we found.
			#print()
			#print("Found: {}".format(tour_length(G, best_solution)))
			#print(best_solution)

			# Print the optimal solution.
			#print()
			#print("Optimal: {}".format(tour_length(G, optimal_tour)))
			#print(optimal_tour)

