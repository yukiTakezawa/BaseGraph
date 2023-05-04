import torch
from dynamic_graph import *
import math


class ExponentialGraph(DynamicGraph):
    def __init__(self, n_nodes):
        w = torch.zeros((n_nodes, n_nodes))

        n_neighbors = int(math.log2(n_nodes-1))
        for i in range(n_nodes):
            w[i,i] = 1 / (math.ceil(math.log2(n_nodes)) + 1)
            
            for j in range(n_neighbors+1):
                w[i, (i+2**j)%n_nodes] = 1 / (math.ceil(math.log2(n_nodes)) + 1)

        super().__init__([w])
