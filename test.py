import networkx as nx
import numpy as np

A = np.array([
[0, 0, 4, 5, 0, 2, 1, 0, 7, 2],
[0, 0, 1, 1, 0, 0, 3, 4, 0, 1],
[4, 1, 0, 0, 0, 0, 0, 1, 0, 1],
[5, 1, 0, 0, 0, 0, 2, 0, 0, 1],
[0, 0, 0, 0, 0, 11, 1, 0, 1, 0],
[2, 0, 0, 0, 11, 0, 1, 3, 3, 12],
[1, 3, 0, 2, 1, 1, 0, 0, 0, 0],
[0, 4, 1, 0, 0, 3, 0, 0, 1, 0],
[7, 0, 0, 0, 1, 3, 0, 1, 0, 0],
[2, 1, 1, 1, 0, 12, 0, 0, 0, 0]])
road_G = nx.from_numpy_matrix(A, create_using=nx.DiGraph)
for src in range(len(A)):
    for dest in range(len(A)):
        if A[src][dest] != 0:
            road_G.edges[src,dest]['length'] = random.randint(20,200)
charging = {"0":1, "1":1, "2":1, "3":1, "4":1, "5":1, "6":1, "7":1, "8":1, "9":1}
miles_per_percent = 5
G = layer_graph(road_G, 50)