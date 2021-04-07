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

    
class LocalUpdateJO(object):
    def __init__(self, args, dataset=None, idxs=None, results=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.dataset = dataset
        
        train_data_lst = []
        for idx in idxs:
            train_data_lst.append(dataset[idx])

        self.ldr_train = DataLoader(train_data_lst, batch_size=self.args.local_bs, shuffle=True)
        self.args = args
        self.results = results

    def train(self, net, g_epoch=0):
        net.train()
        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)

        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            
            for batch_idx, (images, labels, soft_targets, indexs) in enumerate(self.ldr_train):
                images, labels, soft_targets = images.to(self.args.device), labels.to(self.args.device), soft_targets.to(self.args.device)
                outputs = net(images)
                
                probs, loss = self.mycriterion(outputs, soft_targets)
                
                if g_epoch >= self.args.begin:
                    self.dataset.soft_labels[indexs.cpu().detach().numpy().tolist()] = probs.cpu().detach().numpy().tolist()
                
                net.zero_grad()
                loss.backward()
                optimizer.step()
                if self.args.verbose and batch_idx % 10 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(images), len(self.ldr_train.dataset),
                               100. * batch_idx / len(self.ldr_train), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
            
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss), self.results, self.dataset
    
    
    def mycriterion(self, outputs, soft_targets):
        # We introduce a prior probability distribution p, which is a distribution of classes among all training data.
        p = torch.ones(10).cuda() / 10

        probs = F.softmax(outputs, dim=1)
        avg_probs = torch.mean(probs, dim=0)

        L_c = -torch.mean(torch.sum(F.log_softmax(outputs, dim=1) * soft_targets, dim=1))
        L_p = -torch.sum(torch.log(avg_probs) * p)
        L_e = -torch.mean(torch.sum(F.log_softmax(outputs, dim=1) * probs, dim=1))

        loss = L_c + self.args.alpha * L_p + self.args.beta * L_e
        return probs, loss
