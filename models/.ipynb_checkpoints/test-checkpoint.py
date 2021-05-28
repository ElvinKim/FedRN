#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @python: 3.6

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np


def test_img(net_g, data_loader, args):
    net_g.eval()
    test_loss = 0
    correct = 0
    n_total = len(data_loader.dataset)

    for idx, (data, target) in enumerate(data_loader):
        if args.gpu != -1:
            data, target = data.to(args.device), target.to(args.device)
        if args.feature_return:
            log_probs, _ = net_g(data)
        else:
            log_probs = net_g(data)
        # sum up batch loss
        test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
        # get the index of the max log-probability
        y_pred = log_probs.data.max(1, keepdim=True)[1]
        correct += y_pred.eq(target.data.view_as(y_pred)).float().sum().item()

    test_loss /= n_total
    accuracy = 100.0 * correct / n_total
    if args.verbose:
        print('\nTest set: Average loss: {:.4f} \nAccuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, n_total, accuracy))
    return accuracy, test_loss


def test_img_HS(net_g, datatest, args):
    net_g.eval()
    # testing
    test_loss = 0
    correct = 0
    data_loader = DataLoader(datatest, batch_size=args.bs)
    l = len(data_loader)
    for idx, (data, target, soft_label, prediction, index) in enumerate(data_loader):
        if args.gpu != -1:
            data, target, soft_label = data.to(args.device), target.to(args.device), soft_label.to(args.device)
        log_probs = net_g(data)
        # sum up batch loss
        test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
        # get the index of the max log-probability
        y_pred = log_probs.data.max(1, keepdim=True)[1]
        correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

    test_loss /= len(data_loader.dataset)
    accuracy = 100.00 * correct / len(data_loader.dataset)
    if args.verbose:
        print('\nTest set: Average loss: {:.4f} \nAccuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(data_loader.dataset), accuracy))
    return accuracy, test_loss


def test_img_local_all(net_local_list, args, dataset_test, dict_users_test, return_all=False):
    acc_test_local = np.zeros(args.num_users)
    loss_test_local = np.zeros(args.num_users)
    for idx in range(args.num_users):
        net_local = net_local_list[idx]
        net_local.eval()
        a, b = test_img_local(net_local, dataset_test, args, user_idx=idx, idxs=dict_users_test[idx])

        acc_test_local[idx] = a
        loss_test_local[idx] = b

    if return_all:
        return acc_test_local, loss_test_local
    return acc_test_local.mean(), loss_test_local.mean()
