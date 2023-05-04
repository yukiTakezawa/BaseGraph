import torch
from dynamic_graph import *
import math


class OnePeerExponentialGraph(DynamicGraph):
    def __init__(self, n_nodes):
        w_list = []

        n_neighbors = int(math.log2(n_nodes-1))

        for j in range(n_neighbors+1):
            
            w = torch.zeros((n_nodes, n_nodes))
            for i in range(n_nodes):
                w[i,i] = 1/2
                w[i, (i+2**j)%n_nodes] = 1/2
                
            w_list.append(w)
            
        super().__init__(w_list)
