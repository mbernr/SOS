from tsp_parser import parse_tsp
from utils import get_path_length


edgelist = "tsp_instances/0010_k1.txt"
coords = "tsp_instances/kroB150_k3_2.txt"
G = parse_tsp(edgelist)

print(get_path_length(G, [0,4,7,1]))