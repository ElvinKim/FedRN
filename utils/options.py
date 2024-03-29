#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse


def args_parser():
    parser = argparse.ArgumentParser()
    # label noise method
    parser.add_argument('--method', type=str, default='default',
                        choices=['default', 'selfie', 'jointoptim', 'coteaching', 'coteaching+', 'dividemix', 'fedrn'],
                        help='method name')

    # federated arguments
    parser.add_argument('--epochs', type=int, default=500, help="rounds of training")
    parser.add_argument('--num_users', type=int, default=100, help="number of users: K")
    parser.add_argument('--frac', type=float, default=0.1, help="the fraction of clients: C")
    parser.add_argument('--local_ep', type=int, default=5, help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=50, help="local batch size: B")
    parser.add_argument('--bs', type=int, default=128, help="test batch size")
    parser.add_argument('--lr', type=float, default=0.01, help="learning rate")
    parser.add_argument('--momentum', type=float, default=0.5, help="SGD momentum (default: 0.5)")
    parser.add_argument('--split', type=str, default='user', help="train-test split type, user or sample")
    parser.add_argument('--schedule', nargs='+', default=[], help='decrease learning rate at these epochs.')
    parser.add_argument('--lr_decay', type=float, default=0.1, help="learning rate decay")
    parser.add_argument('--weight_decay', type=float, default=0, help="sgd weight decay")
    parser.add_argument('--partition', type=str, choices=['shard', 'dirichlet'], default='shard')
    parser.add_argument('--dd_alpha', type=float, default=0.5, help="dirichlet distribution alpha")
    parser.add_argument('--num_shards', type=int, default=200, help="number of shards")
    parser.add_argument('--fed_method', type=str, default='fedavg', choices=['fedavg'],
                        help="federated learning method")

    # model arguments
    parser.add_argument('--model', type=str, default='cnn4conv', choices=['cnn4conv'], help='model name')

    # other arguments
    parser.add_argument('--dataset', type=str, default='cifar10', help="name of dataset",
                        choices=['cifar10', 'cifar100'])
    parser.add_argument('--iid', action='store_true', help='whether i.i.d or not')
    parser.add_argument('--num_classes', type=int, default=10, help="number of classes")
    parser.add_argument('--num_channels', type=int, default=3, help="number of channels of imges")
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")
    parser.add_argument('--verbose', action='store_true', help='verbose print')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--all_clients', action='store_true', help='aggregation over all clients')
    parser.add_argument('--num_workers', type=int, default=4, help='num_workers to load data')

    # noise label arguments
    parser.add_argument('--noise_type_lst', nargs='+', default=['symmetric'], help='[pairflip, symmetric]')
    parser.add_argument('--noise_group_num', nargs='+', default=[100], type=int)
    parser.add_argument('--group_noise_rate', nargs='+', default=[0.2], type=float,
                        help='Should be 2 noise rates for each group: min_group_noise_rate max_group_noise_rate but '
                             'if there is only 1 group and 1 noise rate, same noise rate will be applied to all users')
    parser.add_argument('--warmup_epochs', type=int, default=100, help='number of warmup epochs')

    # SELFIE / Joint optimization arguments
    parser.add_argument('--queue_size', type=int, default=15, help='size of history queue')
    # SELFIE / Co-teaching arguments
    parser.add_argument('--forget_rate', type=float, default=0.2, help="forget rate for co-teaching")
    # SELFIE arguments
    parser.add_argument('--uncertainty_threshold', type=float, default=0.05, help='uncertainty threshold')
    # Joint optimization arguments
    parser.add_argument('--alpha', type=float, default=1.2, help="alpha for joint optimization")
    parser.add_argument('--beta', type=float, default=0.8, help="beta for joint optimization")
    parser.add_argument('--labeling', type=str, default='soft', help='[soft, hard]')
    # MixMatch arguments
    parser.add_argument('--mm_alpha', default=4, type=float)
    parser.add_argument('--lambda_u', default=25, type=float, help='weight for unsupervised loss')
    parser.add_argument('--T', default=0.5, type=float)
    parser.add_argument('--p_threshold', default=0.5, type=float)

    # FedRN
    parser.add_argument('--num_neighbors', type=int, default=2, help="number of neighbors")
    parser.add_argument('--w_alpha', type=float, help='weight alpha for our method', default=0.5)

    args = parser.parse_args()
    return args
