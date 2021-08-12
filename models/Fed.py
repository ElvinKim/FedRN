#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torch import nn


class LocalModelWeights:
    def __init__(self, all_clients, net_glob, num_users, method, dict_users):
        self.all_clients = all_clients
        self.num_users = num_users
        self.method = method
        self.user_data_size = [len(dict_users[i]) for i in range(num_users)]
        # if the data size for each user is the same,
        # set all elements in user_data_size as 1
        if self.user_data_size and \
                all([self.user_data_size[0] == data_size for data_size in self.user_data_size]):
            self.user_data_size = [1] * len(self.user_data_size)
        
        w_glob = net_glob.state_dict()
        if self.all_clients:
            print("Aggregation over all clients")
            self.w_locals = [w_glob for i in range(self.num_users)]
            self.data_size_locals = self.user_data_size
        else:
            self.w_locals = []
            self.data_size_locals = []

    def init(self):
        # Reset local weights if necessary
        if not self.all_clients:
            self.w_locals = []
            self.data_size_locals = []

    def update(self, idx, w):
        if self.all_clients:
            self.w_locals[idx] = copy.deepcopy(w)
        else:
            self.w_locals.append(copy.deepcopy(w))
            self.data_size_locals.append(self.user_data_size[idx])

    def average(self):
        w_glob = None
        if self.method == 'fedavg':
            w_glob = FedAvg(self.w_locals, self.data_size_locals)

        return w_glob


def FedAvg(w, average_weights):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        w_avg[k] *= average_weights[0]
        for i in range(1, len(w)):
            w_avg[k] += w[i][k] * average_weights[i]
        w_avg[k] = torch.div(w_avg[k], sum(average_weights))
            
    return w_avg
