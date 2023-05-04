import torch
import math
import re
import copy
import random
import sympy

from dynamic_graph import *
from vis_simple_adic_graph import *



class VisAdicGraph(DynamicGraph):
    def __init__(self, n_nodes, max_degree=1, seed=0, inner_edges=True):
        self.state = np.random.RandomState(seed)
        self.inner_edges = inner_edges
        self.max_degree = max_degree
        self.n_nodes = n_nodes
        self.seed = seed

        super().__init__(self.construct())

    def construct(self):
        node_list_list1, node_list_list2, n_power, n_rest = self.split_nodes()

        simple_adics = [VisSimpleAdicGraph(len(node_list_list1[i]), max_degree=self.max_degree) for i in range(n_power)]
        hyper_cubes = [VisHyperHyperCube(len(node_list_list2[i]), max_degree=self.max_degree) for i in range(n_rest)]

        # check which is better
        g = VisSimpleAdicGraph(self.n_nodes, max_degree=self.max_degree, seed=self.seed, inner_edges=self.inner_edges)
        if len(g.w_list) < len(simple_adics[0].w_list) + len(hyper_cubes[0].w_list):
            return g.w_list


        w_list = []
        for m in range(len(simple_adics[0].w_list)):
            w = torch.zeros((self.n_nodes, self.n_nodes))

            for l in range(n_power):
                w += self.extend(simple_adics[l].w_list[m], node_list_list1[l])
            w_list.append(w)

        for m in range(len(hyper_cubes[0].w_list)):
            w = torch.zeros((self.n_nodes, self.n_nodes))

            for l in range(n_rest):
                w += self.extend(hyper_cubes[l].w_list[m], node_list_list2[l]) * 6.0
            w_list.append(w)

        return w_list


    def diag(self, X, Y):
        new_W = torch.zeros((X.size()[0] + Y.size()[0], X.size()[0] + Y.size()[0]))
        new_W[0:X.size()[0], 0:X.size()[0]] = X
        new_W[X.size()[0]:, X.size()[0]:] = Y
        return new_W


    def extend(self, w, node_list):
        new_w = torch.zeros((self.n_nodes, self.n_nodes))
        for i in range(len(node_list)):
            for j in range(len(node_list)):
                new_w[node_list[i], node_list[j]] = w[i, j]
        return new_w


    def split_nodes(self):
        factors = [n**int(math.log(self.n_nodes, n)) for n in range(2, self.max_degree+2)]
        factor = np.prod(factors)
        n_power = math.gcd(self.n_nodes, factor)
        n_rest = int(self.n_nodes / n_power)

        node_list = list(range(self.n_nodes))
        node_list_list1 = []
        for i in range(n_power):
            node_list_list1.append(node_list[n_rest*i:n_rest*(i+1)])

        node_list_list2 = [[] for _ in range(n_rest)]
        for i in range(n_power):
            for j in range(n_rest):
                node_list_list2[j].append(node_list_list1[i][j])

        return node_list_list1, node_list_list2, n_power, n_rest


    def get_neighbors(self, i):
        in_neighbors = self.get_in_neighbors(i)
        out_neighbors = self.get_out_neighbors(i)
        self.itr += 1

        if self.itr % len(self.w_list) == 0:
            self.w_list = self.shuffle_node_index(self.w_list, self.n_nodes)

        return in_neighbors, out_neighbors


    def shuffle_node_index(self, w_list, n_nodes):
        node_index_list = list(range(n_nodes))
        self.state.shuffle(node_index_list)

        new_w_list = []
        for w in w_list:
            new_w = torch.zeros_like(w)
            for i in range(n_nodes):
                for j in range(n_nodes):
                    new_w[i, j] = w[node_index_list[i], node_index_list[j]]
            new_w_list.append(new_w)
        return new_w_list
