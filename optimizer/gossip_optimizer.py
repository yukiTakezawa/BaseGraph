import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.optim.optimizer import Optimizer
import torch.distributed as dist
import time
import random


class GossipOptimizer(Optimizer):
    def __init__(self, params, node_id: int, graph, local_step, lr=1e-5, beta=0.9, device="cuda"):

        self.node_id = node_id
        self.graph = graph
        self.device = device

        self.local_step = local_step
        self.step_counter = 0
        self.l2_penalty = 0.001
        self.lr = lr
        
        defaults = dict(lr=lr, beta=beta)
        super(GossipOptimizer, self).__init__(params, defaults)

        for group in self.param_groups:
            group["momentum"] = []

            for p in group["params"]:
                group["momentum"].append(torch.zeros_like(p, device=self.device))
            
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        
        for group in self.param_groups:
            lr = group['lr']
            beta = group["beta"]
            
            for p, momentum in zip(group['params'], group["momentum"]):
                momentum.data = p.grad + beta * momentum
                p.data = p.data - lr * momentum
                
        if closure is not None:
            loss = closure()

        self.step_counter += 1
        if self.step_counter % self.local_step == 0:
            self.update()

        return loss


    @torch.no_grad()
    def update(self):

        in_neighbors, out_neighbors = self.graph.get_neighbors(self.node_id)
        
        task_list = []
        recieved_params = {}
        for node_id in out_neighbors.keys():
            if node_id != self.node_id:
                task_list += self.send_param(node_id)

        for node_id in in_neighbors:
            if node_id != self.node_id:
                tasks, params = self.recv_param(node_id)
                task_list += tasks
                recieved_params[node_id] = params
            
        for task in task_list:
            task.wait()
            
        self.average_param(recieved_params, in_neighbors)

        
    @torch.no_grad()
    def send_param(self, node_id):
        task_list = []
        
        for group in self.param_groups:
            for i, p in enumerate(group["params"]):
                task_list.append(dist.isend(tensor=p.to("cpu"), dst=node_id, tag=i))

        return task_list

    
    @torch.no_grad()
    def recv_param(self, node_id):
        task_list = []
        recieved_params = []
        
        for group in self.param_groups:        
            for i, p in enumerate(group["params"]):    
                tmp = torch.zeros_like(p, device="cpu")
                task_list.append(dist.irecv(tensor=tmp, src=node_id, tag=i))
                recieved_params.append(tmp)
                
        return task_list, recieved_params

    
    @torch.no_grad()
    def average_param(self, recieved_params, neighbors): 
                
        for group in self.param_groups:
            for i, p in enumerate(group["params"]):
                p.data *=neighbors[self.node_id]
                
                for node_id in neighbors.keys():
                    if node_id != self.node_id:
                        p.data += neighbors[node_id] * recieved_params[node_id][i].to(self.device)

