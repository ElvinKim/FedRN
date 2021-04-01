#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import numpy as np
import random
from sklearn import metrics


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        item = int(item)
        image, label = self.dataset[self.idxs[item]]
        return image, label

def loss_coteaching(y_pred1, y_pred2, y_true, forget_rate):
    loss_1 = F.cross_entropy(y_pred1, y_true, reduce = False)
    ind_1_sorted = np.argsort(loss_1.data.cpu()).cuda()
    loss_1_sorted = loss_1[ind_1_sorted]

    loss_2 = F.cross_entropy(y_pred2, y_true, reduce = False)
    ind_2_sorted = np.argsort(loss_2.data.cpu()).cuda()
    loss_2_sorted = loss_2[ind_2_sorted]

    remember_rate = 1 - forget_rate
    num_remember = int(remember_rate * len(loss_1_sorted))

    ind_1_update=ind_1_sorted[:num_remember]
    ind_2_update=ind_2_sorted[:num_remember]
    # exchange
    loss_1_update = F.cross_entropy(y_pred1[ind_2_update], y_true[ind_2_update])
    loss_2_update = F.cross_entropy(y_pred2[ind_1_update], y_true[ind_1_update])

    return torch.sum(loss_1_update)/num_remember, torch.sum(loss_2_update)/num_remember
    
    
class LocalUpdate(object):
    def __init__(self, args, dataset=None, idxs=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)
        
    def train(self, net):
        net.train()
        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        
        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                if self.args.verbose and batch_idx % 10 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(images), len(self.ldr_train.dataset),
                               100. * batch_idx / len(self.ldr_train), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)

    
class LocalUpdate_coteaching(object):
    def __init__(self, args, dataset=None, idxs=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)
        
    def train_coteaching(self, net1, net2, rate_schedule):
        net1.train()
        net2.train()
        # train and update
        optimizer1 = torch.optim.SGD(net1.parameters(), lr=self.args.lr, momentum=self.args.momentum, weight_decay=self.args.weight_decay)
        optimizer2 = torch.optim.SGD(net2.parameters(), lr=self.args.lr, momentum=self.args.momentum, weight_decay=self.args.weight_decay)

        epoch_loss1 = []
        epoch_loss2 = []
        for iter in range(self.args.local_ep):
            batch_loss1 = []
            batch_loss2 = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net1.zero_grad()
                net2.zero_grad()
                log_probs1 = net1(images)
                log_probs2 = net2(images)
                loss1, loss2 = loss_coteaching(log_probs1, log_probs2, labels, rate_schedule)
                loss1.backward()
                loss2.backward()
                optimizer1.step()
                optimizer2.step()
                if self.args.verbose and batch_idx % 10 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss1: {:.6f}\tLoss2: {:.6f}'.format(
                        iter, batch_idx * len(images), len(self.ldr_train.dataset),
                               100. * batch_idx / len(self.ldr_train), loss1.item(), loss2.item()))
                batch_loss1.append(loss1.item())
                batch_loss2.append(loss2.item())
            epoch_loss1.append(sum(batch_loss1)/len(batch_loss1))
            epoch_loss2.append(sum(batch_loss2)/len(batch_loss2))
        return net1.state_dict(), sum(epoch_loss1) / len(epoch_loss1), net2.state_dict(), sum(epoch_loss2) / len(epoch_loss2)