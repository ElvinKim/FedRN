#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torch import nn


class LocalModelWeights:
    def __init__(self, all_clients, net_glob, num_users, method, model):
        self.all_clients = all_clients
        self.num_users = num_users
        self.method = method
        self.model = model
        
        w_glob = net_glob.state_dict()
        self.w_locals = []

        if self.all_clients:
            print("Aggregation over all clients")
            self.w_locals = [w_glob for i in range(self.num_users)]

    def init(self):
        # Reset local weights if necessary
        if not self.all_clients:
            self.w_locals = []

    def update(self, idx, w):
        if self.all_clients:
            self.w_locals[idx] = copy.deepcopy(w)
        else:
            self.w_locals.append(copy.deepcopy(w))

    def average(self):
        w_glob = None
        if self.method == 'fedavg':
            w_glob = FedAvg(self.w_locals, self.model)

        return w_glob


def FedAvg(w, model):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
            
    return w_avg
