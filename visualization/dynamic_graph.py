import torch
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


class DynamicGraph():
    def __init__(self, w_list):
        """
        Parameter
        --------
        w_list (list of torch.tensor):
            list of mixing matrix
        """
        self.w_list = w_list
        self.n_nodes = w_list[0].size()[0]
        self.length = len(w_list)
        self.itr = 0
        
    def get_in_neighbors(self, i):
        """
        Parameter
        ----------
        i (int):
            a node index
        Return
        ----------
            dictionary of (neighbors's index: weight of the edge (i,j))
        """
        w = self.w_list[self.itr%self.length]        

        return {idx.item(): w[idx, i].item() for idx in torch.nonzero(w[:,i])}

    def get_out_neighbors(self, i):
        """
        Parameter
        ----------
        i (int):
            a node index
        Return
        ----------
            dictionary of (neighbors's index: weight of the edge (i,j))
        """
        w = self.w_list[self.itr%self.length]        
        
        return {idx.item(): w[i,idx].item() for idx in torch.nonzero(w[i])}

    
    def get_neighbors(self, i):
        in_neighbors = self.get_in_neighbors(i)
        out_neighbors = self.get_out_neighbors(i)
        self.itr += 1
        return in_neighbors, out_neighbors
        
    
    def get_w(self):
        w = self.w_list[self.itr%self.length]        
        self.itr += 1
        return w
    
    def visualize(self):
        
        pos = {str(n): 0.5*np.array([np.cos(2 * np.pi * n / self.n_nodes), np.sin(2 * np.pi * n / self.n_nodes)]) for n in range(self.n_nodes)}
        plt.figure(figsize=(2*self.length,2))
    
        for t in range(self.length):
            plt.subplot(1, self.length, t+1)
            G = nx.Graph()
        
            for i in range(self.n_nodes):
                G.add_node(str(i))
                
                for i in range(self.n_nodes):
                    for j in range(self.n_nodes):
                        if self.w_list[t][i,j] > 0:
                            G.add_edge(str(i), str(j))
                            
            nx.draw(G, pos=pos)
        plt.show()
