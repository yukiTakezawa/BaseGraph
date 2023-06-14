import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.optim.optimizer import Optimizer
import random
from torch.utils.data.dataset import Subset
from more_itertools import chunked
import numpy as np
import math


def distribute_data_dirichlet(
    targets, non_iid_alpha, n_workers, seed=0, num_auxiliary_workers=10
):
    # we refer https://github.com/epfml/relaysgd/tree/89719198ba227ebbff9a6bf5b61cb9baada167fd
    """Code adapted from Tao Lin (partition_data.py)"""
    random_state = np.random.RandomState(seed=seed)

    num_indices = len(targets)
    num_classes = len(np.unique(targets))

    indices2targets = np.array(list(enumerate(targets)))
    random_state.shuffle(indices2targets)

    # partition indices.
    from_index = 0
    splitted_targets = []
    num_splits = math.ceil(n_workers / num_auxiliary_workers)
    split_n_workers = [
        num_auxiliary_workers
        if idx < num_splits - 1
        else n_workers - num_auxiliary_workers * (num_splits - 1)
        for idx in range(num_splits)
    ]
    split_ratios = [_n_workers / n_workers for _n_workers in split_n_workers]
    for idx, ratio in enumerate(split_ratios):
        to_index = from_index + int(num_auxiliary_workers / n_workers * num_indices)
        splitted_targets.append(
            indices2targets[
                from_index : (num_indices if idx == num_splits - 1 else to_index)
            ]
        )
        from_index = to_index

    idx_batch = []
    for _targets in splitted_targets:
        # rebuild _targets.
        _targets = np.array(_targets)
        _targets_size = len(_targets)

        # use auxi_workers for this subset targets.
        _n_workers = min(num_auxiliary_workers, n_workers)
        n_workers = n_workers - num_auxiliary_workers

        # get the corresponding idx_batch.
        min_size = 0
        while min_size < int(0.50 * _targets_size / _n_workers):
            _idx_batch = [[] for _ in range(_n_workers)]
            for _class in range(num_classes):
                # get the corresponding indices in the original 'targets' list.
                idx_class = np.where(_targets[:, 1] == _class)[0]
                idx_class = _targets[idx_class, 0]

                # sampling.
                try:
                    proportions = random_state.dirichlet(
                        np.repeat(non_iid_alpha, _n_workers)
                    )
                    # balance
                    proportions = np.array(
                        [
                            p * (len(idx_j) < _targets_size / _n_workers)
                            for p, idx_j in zip(proportions, _idx_batch)
                        ]
                    )
                    proportions = proportions / proportions.sum()
                    proportions = (np.cumsum(proportions) * len(idx_class)).astype(int)[
                        :-1
                    ]
                    _idx_batch = [
                        idx_j + idx.tolist()
                        for idx_j, idx in zip(
                            _idx_batch, np.split(idx_class, proportions)
                        )
                    ]
                    sizes = [len(idx_j) for idx_j in _idx_batch]
                    min_size = min([_size for _size in sizes])
                except ZeroDivisionError:
                    pass
        idx_batch += _idx_batch
    return idx_batch


## https://github.com/epfml/relaysgd/tree/89719198ba227ebbff9a6bf5b61cb9baada167fd
def dirichlet_split(
        dataset,
        num_workers: int,
        alpha: float = 1,
        seed: int = 0,
        distribute_evenly: bool = True,
    ):
        indices_per_worker = distribute_data_dirichlet(
            dataset.targets, alpha, num_workers, num_auxiliary_workers=10, seed=seed
        )

        if distribute_evenly:
            indices_per_worker = np.array_split(
                np.concatenate(indices_per_worker), num_workers
            )

        return [
            Subset(dataset, indices)
            for indices in indices_per_worker
        ]

    
def load_CIFAR10(n_node, alpha=1.0, batch=128, val_rate=0.2, seed=0):
    """
    node_label : 
        the list of labes that each node has. (example. [[0,1],[1,2],[0,2]] (n_node=3, n_class=2))
    """

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        transforms.RandomErasing(),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    train_val_dataset = datasets.CIFAR10('./data',
                       train=True,
                       download=True,
                       transform=transform_train)
    
    test_dataset = datasets.CIFAR10('./data',
                       train=False,
                       transform=transform_test)


    # split datasets into n_node datasets by Dirichlet distribution. 
    train_val_subset_list = dirichlet_split(train_val_dataset, n_node, alpha, seed=seed)
        
    # the number of train datasets per node.
    n_data = min([len(train_val_subset_list[node_id]) for node_id in range(n_node)])
    n_train = int((1.0 - val_rate) * n_data)
    
    # choose validation datasets.
    val_dataset = None
    train_subset_list = []
    for node_id in range(n_node):
        n_val = len(train_val_subset_list[node_id]) - n_train
        a, b = torch.utils.data.random_split(train_val_subset_list[node_id], [n_train, n_val])
        train_subset_list.append(a)
        
        if val_dataset is None:
            val_dataset = b
        else:
            val_dataset += b
                  
    return {'train': train_subset_list, 'val': val_dataset, 'test': test_dataset}


def load_CIFAR100(n_node, alpha=1.0, batch=128, val_rate=0.2, seed=0):
    """
    node_label : 
        the list of labes that each node has. (example. [[0,1],[1,2],[0,2]] (n_node=3, n_class=2))
    """

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        transforms.RandomErasing(),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    train_val_dataset = datasets.CIFAR100('./data',
                       train=True,
                       download=True,
                       transform=transform_train)
    
    test_dataset = datasets.CIFAR100('./data',
                       train=False,
                       transform=transform_test)

    # split datasets into n_node datasets by Dirichlet distribution. 
    train_val_subset_list = dirichlet_split(train_val_dataset, n_node, alpha, seed=seed)
        
    # the number of train datasets per node.
    n_data = min([len(train_val_subset_list[node_id]) for node_id in range(n_node)])
    n_train = int((1.0 - val_rate) * n_data)
    
    # choose validation datasets.
    val_dataset = None
    train_subset_list = []
    for node_id in range(n_node):
        n_val = len(train_val_subset_list[node_id]) - n_train
        a, b = torch.utils.data.random_split(train_val_subset_list[node_id], [n_train, n_val])
        train_subset_list.append(a)
        
        if val_dataset is None:
            val_dataset = b
        else:
            val_dataset += b
                  
    return {'train': train_subset_list, 'val': val_dataset, 'test': test_dataset}


def load_FMNIST(n_node, alpha=1.0, val_rate=0.2, seed=0):
    """
    node_label : 
        the list of labes that each node has. (example. [[0,1],[1,2],[0,2]] (n_node=3, n_class=2))
    """

    train_val_dataset = datasets.FashionMNIST('./data',
                       train=True,
                       download=True,
                       transform=transforms.Compose([
                           transforms.RandomCrop(28, padding=4),
                           transforms.ToTensor()
                       ]))

    test_dataset = datasets.FashionMNIST('./data',
                       train=False,
                       transform=transforms.Compose([
                           transforms.ToTensor()
                       ]))
    
    # split datasets into n_node datasets by Dirichlet distribution. 
    train_val_subset_list = dirichlet_split(train_val_dataset, n_node, alpha, seed=seed)
        
    # the number of train datasets per node.
    n_data = min([len(train_val_subset_list[node_id]) for node_id in range(n_node)])
    n_train = int((1.0 - val_rate) * n_data)
    
    # choose validation datasets.
    val_dataset = None
    train_subset_list = []
    for node_id in range(n_node):
        n_val = len(train_val_subset_list[node_id]) - n_train
        a, b = torch.utils.data.random_split(train_val_subset_list[node_id], [n_train, n_val])
        train_subset_list.append(a)
        
        if val_dataset is None:
            val_dataset = b
        else:
            val_dataset += b
                  
    return {'train': train_subset_list, 'val': val_dataset, 'test': test_dataset}
                                

def datasets_to_loaders(datasets, batch_size=128):
    """
    datasets dict:
    """
    train_loader = torch.utils.data.DataLoader(
        datasets["train"],
        batch_size=batch_size,
        shuffle=True, num_workers=2, pin_memory=False)
    
    val_loader = torch.utils.data.DataLoader(
        datasets["val"],
        batch_size=batch_size,
        shuffle=False, num_workers=2, pin_memory=False)

    test_loader = torch.utils.data.DataLoader(
        datasets["test"],
        batch_size=batch_size,
        shuffle=False, num_workers=2, pin_memory=False)

    return {"train": train_loader, "val": val_loader, "test": test_loader}
