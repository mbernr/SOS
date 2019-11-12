import sys
import math
import tsplib95
import networkx as nx

### GENERAL GRAPH HELPERS ###

def path_length(G, path):
	path_length = 0
	for i in range(len(path)-1):
		weight = G.get_edge_data(path[i], path[i+1])['weight']
		path_length += weight
	return path_length

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


### TSP HELPERS ###

# Take instances from here: http://elib.zib.de/pub/mp-testdata/tsp/tsplib/tsp/index.html

def parse_tsp(file_path):
	problem = tsplib95.load_problem(file_path)
	G = problem.get_graph()
	G = nx.convert_node_labels_to_integers(G) # otherwise there are problems with the genetic algorithm
	return G

def parse_tsp_optimal_solution(file_path):
	solution = tsplib95.load_solution(file_path)
	tour = [x - 1 for x in solution.tours[0]] # relabel the vertices to range from 0 to n-1, instead of from 1 to n.
	return tour

def tour_length(G, tour):
	path = tour + [tour[0]]
	return path_length(G, path)

def start_tour_from_zero(tour):
	i = tour.index(0)
	return tour[i:] + tour[:i]