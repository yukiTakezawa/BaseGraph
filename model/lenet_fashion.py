import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.optim.optimizer import Optimizer


class LeNetFashion(torch.nn.Module):
    def __init__(self, device="cuda"):
        super(LeNetFashion, self).__init__()
        self.device = device

        self.conv1 = nn.Conv2d(1, 6, 5)
        self.gn1 = nn.GroupNorm(2, 6)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.gn2 = nn.GroupNorm(4, 16)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(256, 120)  # 5*5 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        
        self.criterion = nn.CrossEntropyLoss()

        
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = self.gn1(x)
        # If the size is a square, you can specify with a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = self.gn2(x)
        x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def _calc_acc(self, output, target):
        _, prediction = torch.max(output.data, 1)
        return (prediction == target).sum().item()

    def run(self, loaders, optimizer):
        
        train_loss_list = []
        val_loss_list = []
        train_acc_list = []
        val_acc_list = []
        
        round_cnt = 1
        train_loss = 0
        train_acc = 0
        n_data = 0
        
        for i, (data, target) in enumerate(loaders["train"]):
            self.train()
            self.train()

            optimizer.zero_grad()
            
            data = data.to(self.device)
            target = target.to(self.device)
            output = self.forward(data)
            loss = self.criterion(output, target)

            loss.backward()
            optimizer.step()

            n_data += target.size(0)
            train_loss += loss.detach().cpu().item() * target.size(0)
            train_acc += self._calc_acc(output, target)
            
            #if round_cnt == optimizer.itr_per_round*self.eval_per_round:
            #    val_loss, val_acc = self.run_val(loaders)
            #    val_loss_list.append(val_loss)
            #    val_acc_list.append(val_acc)
            #    train_loss_list.append(train_loss/n_data)
            #    train_acc_list.append(train_acc/n_data)
            #
            #     round_cnt, train_loss, train_acc, n_data = 0, 0, 0, 0
            round_cnt += 1
        #return train_loss_list, train_acc_list, val_loss_list, val_acc_list
        
        return train_loss/n_data, train_acc/n_data



    def run_val(self, loaders):
        self.eval()

        total_loss = 0.
        total_acc = 0.
        n_data = 0 # number of training/validation data.

        with torch.no_grad():
            for i, (data, target) in enumerate(loaders["val"]):
                data = data.to(self.device)
                target = target.to(self.device)
                output = self.forward(data)
                loss = self.criterion(output, target)
                total_loss += loss.item() * target.size(0)
                total_acc += self._calc_acc(output, target)
                n_data += target.size(0)
            
        return total_loss/n_data, total_acc/n_data

    def run_all_train(self, loaders):
        self.eval()

        total_loss = 0.
        total_acc = 0.
        n_data = 0 # number of training/validation data.

        with torch.no_grad():
            for i, (data, target) in enumerate(loaders["all_train"]):
                data = data.to(self.device)
                target = target.to(self.device)
                output = self.forward(data)
                loss = self.criterion(output, target)
                total_loss += loss.item() * target.size(0)
                total_acc += self._calc_acc(output, target)
                n_data += target.size(0)
            
        return total_loss/n_data, total_acc/n_data


    """
    def run_debug(self, loaders, optimizer):
        
        train_loss_list = []
        val_loss_list = []
        train_acc_list = []
        val_acc_list = []
        
        round_cnt = 1
        train_loss = 0
        train_acc = 0
        n_data = 0

        with torch.no_grad():
            for i, (data, target) in enumerate(loaders["train"]):
            
                data = data.to(self.device)
                target = target.to(self.device)
                output = self.forward(data)
                loss = self.criterion(output, target)

                n_data += target.size(0)
                train_loss += loss.detach().cpu().item() * target.size(0)
                train_acc += self._calc_acc(output, target)
        
        return train_loss/n_data, train_acc/n_data
    """
    
    def run_test(self, loaders):
        self.eval()

        total_loss = 0.
        total_acc = 0.
        n_data = 0 # number of training/validation data.
        
        for i, (data, target) in enumerate(loaders["test"]):
            data = data.to(self.device)
            target = target.to(self.device)
            output = self.forward(data)
            loss = self.criterion(output, target)
            total_loss += loss.item() * target.size(0)
            total_acc += self._calc_acc(output, target)
            n_data += target.size(0)

        return total_loss/n_data, total_acc/n_data
