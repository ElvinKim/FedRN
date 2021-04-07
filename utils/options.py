#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse
import sys


def args_parser():
    parser = argparse.ArgumentParser()
    # federated arguments
    parser.add_argument('--epochs', type=int, default=10, help="rounds of training")
    parser.add_argument('--num_users', type=int, default=100, help="number of users: K")
    parser.add_argument('--frac', type=float, default=0.1, help="the fraction of clients: C")
    parser.add_argument('--local_ep', type=int, default=5, help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=10, help="local batch size: B")
    parser.add_argument('--bs', type=int, default=128, help="test batch size")
    parser.add_argument('--lr', type=float, default=0.01, help="learning rate")
    parser.add_argument('--momentum', type=float, default=0.5, help="SGD momentum (default: 0.5)")
    parser.add_argument('--split', type=str, default='user', help="train-test split type, user or sample")
    parser.add_argument('--schedule', nargs='+', default=[50, 80], help='decrease learning rate at these epochs.')
    parser.add_argument('--lr_decay', type=float, default=0.1, help="learning rate decay")
    parser.add_argument('--weight_decay', type=float, default=0, help="sgd weight decay")

    # model arguments
    parser.add_argument('--model', type=str, default='mlp', help='model name')
    parser.add_argument('--kernel_num', type=int, default=9, help='number of each kind of kernel')
    parser.add_argument('--kernel_sizes', type=str, default='3,4,5',
                        help='comma-separated kernel size to use for convolution')
    parser.add_argument('--norm', type=str, default='batch_norm', help="batch_norm, layer_norm, or None")
    parser.add_argument('--num_filters', type=int, default=32, help="number of filters for conv nets")
    parser.add_argument('--max_pool', type=str, default='True',
                        help="Whether use max pooling rather than strided convolutions")
    parser.add_argument('--reproduce', action='store_true', help = 'reproduce paper code')
    
    # other arguments
    parser.add_argument('--dataset', type=str, default='mnist', help="name of dataset")
    parser.add_argument('--iid', action='store_true', help='whether i.i.d or not')
    parser.add_argument('--num_classes', type=int, default=10, help="number of classes")
    parser.add_argument('--num_channels', type=int, default=3, help="number of channels of imges")
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")
    parser.add_argument('--stopping_rounds', type=int, default=10, help='rounds of early stopping')
    parser.add_argument('--verbose', action='store_true', help='verbose print')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--all_clients', action='store_true', help='aggregation over all clients')
    
    # noise label arguments
    parser.add_argument('--noise_type', type = str, help='[pairflip, symmetric]', default='pairflip')
    parser.add_argument('--noise_rate', type = float, help = 'corruption rate, should be less than 1', default = 0.2)
    
    # co-teaching arguments
    parser.add_argument('--num_gradual', type = int, help = 'T_k', default = 10)
    
    # save arguments
    parser.add_argument('--save_dir', type=str, default=None, help="name of save directory")
    
    # Joint Optimization arguments
    if sys.argv[0] == "main_fed_LNL_JointOpt.py":
        parser.add_argument('--alpha', type=float, default=1.2, help="alpha for joint optimization")
        parser.add_argument('--beta', type=float, default=0.8, help="beta for joint optimization")
        parser.add_argument('--begin', type=int, default=35, help='When to begin updating labels')
        
    # Joint Optimization arguments
    if sys.argv[0] in ["main_fed_LNL_diff_NR.py", 'main_fed_diff_NR_JointOpt.py']:
        parser.add_argument('--noise_group_num', nargs='+', default=[50, 50], type=int)
        parser.add_argument('--group_noise_rate', nargs='+', default=[0.1, 0.3], type=float)
        
        parser.add_argument('--alpha', type=float, default=1.2, help="alpha for joint optimization")
        parser.add_argument('--beta', type=float, default=0.8, help="beta for joint optimization")
        parser.add_argument('--begin', type=int, default=35, help='When to begin updating labels')
    
    args = parser.parse_args()
    return args
