import sys
import math
import tsplib95
import networkx as nx
import numpy as np
import acopy
import time

### GENERAL GRAPH HELPERS ###

def path_length(G, path):
	path_length = 0
	for i in range(len(path)-1):
		weight = G.get_edge_data(path[i], path[i+1])['weight']
		path_length += weight
	return path_length

def tour_length(G, tour):
	path = tour + [tour[0]]
	return path_length(G, path)

def start_tour_from_zero(tour):
	i = tour.index(0)
	return tour[i:] + tour[:i]

def max_edge_weight(G):
	max_weight = 0
	for edge in G.edges.data('weight'):
		weight = edge[2]
		if weight > max_weight:
			max_weight = weight
	return max_weight

def edge_weight_sum(G):
	weight_sum = 0
	for edge in G.edges.data('weight'):
		weight = edge[2]
		weight_sum += weight
	return weight_sum

def add_dummy_edges(G):
	M = big_M(G)
	for i in range(G.number_of_nodes()):
		for j in range(G.number_of_nodes()):
			if i != j and not G.has_edge(i,j):
				G.add_edge(i, j, weight=M)
	return G

def big_M(G):
	return edge_weight_sum(G) + 1 


### TSP HELPERS ###

# Take instances from here: http://elib.zib.de/pub/mp-testdata/tsp/tsplib/tsp/index.html

def parse_tsp(file_path):
	problem = tsplib95.load_problem(file_path)
	G = problem.get_graph()
	G = nx.convert_node_labels_to_integers(G) # otherwise there are problems with the genetic algorithm
	if not problem.is_complete():
		G = add_dummy_edges(G)
	return G

def parse_tsp_optimal_solution(file_path):
	solution = tsplib95.load_solution(file_path)
	tour = [x - 1 for x in solution.tours[0]] # relabel the vertices to range from 0 to n-1, instead of from 1 to n.
	return tour


### KNAPSACK HELPERS ###

class KnapsackInstance:
	def __init__(self, capacity, number_items, weights_items, values_items):
		self.capacity = capacity			# (int)            maximum load of the knapsack
		self.number_items = number_items	# (int)            number of candidate items
		self.weights_items = weights_items	# (array of int)   weights of the items (weights_items[i]=weight of item i)
		self.values_items = values_items	# (array of int)   value of the items (value_items[i]=value of item i)
		self.pheromones = np.zeros(number_items, dtype=float)
		self.ub = sum(values_items)			# upper bound on the knapsack value

# return KnapsackInstance object defined above
def parse_knapsack(file_path):
	with open(file_path, 'r') as file:
		number_items, capacity = (int(el) for el in file.readline().split())
		weights = []
		values = []
		for i in range(0,number_items):
			value, weight = (int(el) for el in file.readline().split())
			weights.append(weight)
			values.append(value)

		return KnapsackInstance(capacity, number_items, weights, values)

# return optimal knapsack value (int)
def parse_knapsack_optimal_solution(file_path):
	with open(file_path, 'r') as file:
		return int(file.readline())

### ACOPY plugin for displaying stats at each iteration

class StatsIterations(acopy.solvers.SolverPlugin):

	def __init__(self, evalFunction):
		super().__init__(evalFunction=evalFunction)
		self.evalFunction = evalFunction
		self.colwidth = 15
		print("{:<5} {:<15} {:<15} {:<15} {:<15} {:<15}".format("iter", "best", "avg", "std", "min", "max"))
		self.counter = 0

	def on_iteration(self, state):
		self.counter += 1
		evals = [self.evalFunction(solution) for solution in state.solutions]

		print("{:<5} {:<15} {:<15.2f} {:<15.2f} {:<15} {:<15}".format(
			self.counter, 
			self.evalFunction(state.record),
			np.mean(evals), 
			np.std(evals),
			np.min(evals),
			np.max(evals))
		)