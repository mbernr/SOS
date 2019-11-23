import random
import numpy as np
from deap import algorithms, base, creator, tools
from utils import *
import acopyKS

random.seed(0)



#ANTS ALGORITHM PARAMETERS
#influence parameters: define wether the pheromones or the edge distance is more important for choosing an edge for an ant
pheromone_influence = 3
distance_influence = 1
rho = 0.03 				# percentage of pheromones that evaporate after each iteration
q = 1 					# amount of pheromone each ant can deposit
ants_count = 20			# if set to None => one ant per node
number_iter = 10


print("{},{},{},{},{},{},{}".format("beta","alpha","nAnts","evap","time","best_sol","score"))
for ab in [(4,4)]:
	for ants_count in [100]:
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
			G = parse_knapsack("instances/knapsack/large_scale/knapPI_3_1000_1000_1")
			optimal_value = parse_knapsack_optimal_solution("instances/knapsack/large_scale-optimum/knapPI_3_1000_1000_1")

			# Creating Acopy Solver
			solver = acopyKS.Solver(rho=rho, q=q)
			colony = acopyKS.Colony(alpha=distance_influence, beta=pheromone_influence)
			time_limit = acopy.plugins.TimeLimit(seconds=30)
			solver.add_plugin(time_limit)

			# Perform algorithm
			solution = solver.solve(G, colony, gen_size=ants_count, limit=number_iter)

			# retrieve solution
			best_solution = solution.value
			
			end = time.time() - start

			print("{},{},{},{},{},{},{}".format(ab[0], ab[1], ants_count, pheromone_evap, end, best_solution, best_solution/optimal_value))

