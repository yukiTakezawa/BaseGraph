import torch
from dynamic_graph import *


class Ring(DynamicGraph):
    def __init__(self, n_nodes):
        w = torch.zeros((n_nodes, n_nodes))

        for i in range(n_nodes):
            w[i,i] = 1/3
            w[i, (i+1)%n_nodes] = 1/3
            w[i, (i-1)%n_nodes] = 1/3

        super().__init__([w])
