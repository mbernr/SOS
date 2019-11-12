
def path_length(G, path):
	path_length = 0
	for i in range(len(path)-1):
		weight = G.get_edge_data(path[i], path[i+1])['weight']
		path_length += weight
	return path_length