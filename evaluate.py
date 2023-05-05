import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import time
from collections import defaultdict, OrderedDict
import pickle
from tqdm import tqdm
import random
import math
import argparse    
import json
import random
import numpy as np
from timm.scheduler import *

from model.vgg_cifar10 import *
from model.vgg_cifar100 import*
from model.lenet_fashion import *

from optimizer.gossip_optimizer import *
from optimizer.d2_optimizer import *

from data.loader_dirichlet import *

from ring import *
from exponential_graph import *
from one_peer_exponential_graph import *
from base_graph import *


torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def run(rank, size, datasets, config):
    # initialize the model parameters with same seed value.
    torch.manual_seed(config["seed"])
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    
    torch.set_num_threads(1)

    if config["model"] == "vgg":
        if config["dataset"] == "cifar10":
            net = VggCifar10(device=config["device"][rank]).to(config["device"][rank])
        elif config["dataset"] == "cifar100":
            net = VggCifar100(device=config["device"][rank]).to(config["device"][rank])            
    if config["model"] == "lenet":
        net = LeNetFashion(device=config["device"][rank]).to(config["device"][rank])            
            
    net.to(config["device"][rank])
    
    loaders = datasets_to_loaders(datasets, config["batch"])

    if config["optimizer"] == "gossip":
        optimizer = GossipOptimizer(params=net.parameters(), node_id=rank, graph=config["graph"], local_step=config["local_step"], lr=config["lr"], beta=config["beta"], device=config["device"][rank])
    elif config["optimizer"] == "d2":
        optimizer = D2Optimizer(params=net.parameters(), node_id=rank, graph=config["graph"], local_step=config["local_step"], lr=config["lr"], beta=config["beta"], device=config["device"][rank])

    scheduler = CosineLRScheduler(optimizer, t_initial=config["epochs"], lr_min=1e-4, warmup_t=10, warmup_lr_init=5e-5, warmup_prefix=True)
    
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "test_loss": [], "test_acc": [], "diff_param": []}
    count_epoch = 0
    
    with tqdm(range(config["epochs"]), desc=("node "+str(rank)), position=rank) as pbar:
        for epoch in pbar:
            
            train_loss, train_acc = net.run(loaders, optimizer)

            if (count_epoch % 10 == 0) or (count_epoch == config["epochs"] -1):
                val_loss, val_acc = net.run_val(loaders)
                test_loss, test_acc = net.run_test(loaders)
                
                # save loss and accuracy
                history["train_loss"] += [train_loss]
                history["test_loss"] += [test_loss]
                history["val_loss"] += [val_loss]
                history["train_acc"] += [train_acc]
                history["test_acc"] += [test_acc]
                history["val_acc"] += [val_acc]
                        
                pbar.set_postfix(OrderedDict(loss=(round(train_loss, 2), round(test_loss, 2)), acc=(round(train_acc, 2), round(test_acc, 2))))
                
            count_epoch += 1
            scheduler.step(count_epoch)
                
    pickle.dump(history, open(config["log_path"] + "node" + str(rank) + ".pk", "wb"))
    
    
def init_process(rank, size, datasets, config, fn, backend='gloo'):
    os.environ['MASTER_ADDR'] = config["config"]["master_address"] #'127.0.0.1'
    os.environ['MASTER_PORT'] = config["config"]["port"]
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size, datasets, config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('log', default="./results", type=str, help="A directory where the log is stored.")
    parser.add_argument('--dataset', default="cifar10", type=str, help="Dataset. cifar10 and cifar100 are available.") 
    parser.add_argument('--optimizer', default="gossip", type=str, help="Optimization method.")
    parser.add_argument('--batch', default=32, type=int, help="Batch size.")
    parser.add_argument('--seed', default=0, type=int, help="Seed value.")
    parser.add_argument('--model', default="lenet", type=str, help="Neural network architecture. vgg is available.")
    parser.add_argument('--nw', default="one_peer_base", type=str, help="An undelying network topology. one_peer_base, two_peer_base, etc. are available.")
    parser.add_argument('--cuda', default=None, type=str)
    parser.add_argument('--config', default="config/8_node.json", type=str, help="A configulation file.")
    parser.add_argument('--node_list', nargs="*", type=int, help="A list of node id.")
    parser.add_argument('--lr', default=1e-3, type=float, help="Learning rate.")
    parser.add_argument('--epoch', default=1000, type=int, help="The number of epochs.")
    parser.add_argument('--alpha', default=100, type=float, help="Hyperparameter of Dirichlet distribution.")
    parser.add_argument('--beta', default=0.9, type=float, help="Momentum coefficient.")
    parser.add_argument('--local_step', default=5, type=int, help="The number of local step. The local step is not available now. Please specify 1.")
    args = parser.parse_args()

    config = defaultdict(dict)
    config["dataset"] = args.dataset
    config["optimizer"] = args.optimizer
    config["lr"] = args.lr
    config["seed"] = args.seed 
    config["epochs"] = args.epoch
    config["log_path"] = args.log
    config["batch"] = args.batch
    config["model"] = args.model
    config["node_list"] = args.node_list
    config["nw"] = args.nw
    config["beta"] = args.beta
    config["alpha"] = args.alpha
    config["local_step"] = args.local_step
    
    with open(args.config) as f:
        config["config"] = json.load(f)
    
    n_nodes = config["config"]["n_nodes"]
    
    if config["nw"] == "ring":
        config["graph"] = Ring(n_nodes)
    elif config["nw"] == "exp":
        config["graph"] = ExponentialGraph(n_nodes)
    elif config["nw"] == "one_peer_exp":
        config["graph"] = OnePeerExponentialGraph(n_nodes)
    elif config["nw"] == "one_peer_base":
        config["graph"] = BaseGraph(n_nodes, max_degree=1, seed=config["seed"])
    elif config["nw"] == "two_peer_base":
        config["graph"] = BaseGraph(n_nodes, max_degree=2, seed=config["seed"])
    elif config["nw"] == "three_peer_base":
        config["graph"] = BaseGraph(n_nodes, max_degree=3, seed=config["seed"])
    elif config["nw"] == "four_peer_base":
        config["graph"] = BaseGraph(n_nodes, max_degree=4, seed=config["seed"])
    else:
        print("ERROR: exp, one_peer_exp, {one,two,three,four}_peer_base are available", file=sys.stderr)
        sys.exit(1)

    if args.cuda is None:
        config["device"] = {node_id : config["config"][f"node{node_id}"]["cuda"] for node_id in config["node_list"]}
    else:
        config["device"] = [args.cuda for _ in range(n_nodes)]

    torch.manual_seed(config["seed"])
    random.seed(config["seed"])
    np.random.seed(config["seed"])

    if config["dataset"] == "cifar10":
        datasets = load_CIFAR10(n_nodes, batch=config["batch"], alpha=config["alpha"], val_rate=0.1, seed=config["seed"])
    elif config["dataset"] == "cifar100":
        datasets = load_CIFAR100(n_nodes, batch=config["batch"], alpha=config["alpha"], val_rate=0.1, seed=config["seed"])
    elif config["dataset"] == "fashion_mnist":
        datasets = load_FMNIST(n_nodes, alpha=config["alpha"], val_rate=0.1, seed=config["seed"])        
    else:
        print('cifar10 or cifar100 are available in dataset', file=sys.stderr)
        sys.exit(1)

        
    processes = []
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass
    
    for rank in config["node_list"]:
        print(rank)
        node_datasets = {"train": datasets["train"][rank], "val": datasets["val"], "test": datasets["test"]}
        p = mp.Process(target=init_process, args=(rank, n_nodes, node_datasets, config, run))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
