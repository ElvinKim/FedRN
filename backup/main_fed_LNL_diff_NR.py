#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms
import torch

from utils.sampling import mnist_iid, mnist_noniid, cifar_iid, cifar_noniid
from utils.options import args_parser
from utils.utils import noisify_label
from models.Update import LocalUpdate
from models.Nets import MLP, CNNMnist, CNNCifar
from models.Fed import FedAvg
from models.test import test_img

import csv
import os


if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    
    # Seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(args.seed)

    # load dataset and split users
    if args.dataset == 'mnist':
        from six.moves import urllib    
        opener = urllib.request.build_opener()
        opener.addheaders = [('User-agent', 'Mozilla/5.0')]
        urllib.request.install_opener(opener)
        
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('./data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('./data/mnist/', train=False, download=True, transform=trans_mnist)
        # sample users
        if args.iid:
            dict_users = mnist_iid(dataset_train, args.num_users)
        else:
            dict_users = mnist_noniid(dataset_train, args.num_users)
    elif args.dataset == 'cifar':
        trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10('./data/cifar', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR10('./data/cifar', train=False, download=True, transform=trans_cifar)
        
        if args.iid:
            dict_users = cifar_iid(dataset_train, args.num_users)
        else:
            dict_users = cifar_noniid(dataset_train, args.num_users)
    else:
        exit('Error: unrecognized dataset')
    img_size = dataset_train[0][0].shape
    
        
    # Insert Noise
    train_dataset = []

    for d in dataset_train:
        train_dataset.append([d[0], d[1]])

    if sum(args.noise_group_num) != args.num_users:
        exit('Error: sum of the number of noise group have to be equal the number of users')

    group_idx_lst = []
    prev_idx = 0

    for g_num in args.noise_group_num:
        group_idx = g_num + prev_idx
        group_idx_lst.append([prev_idx, group_idx])
        prev_idx = group_idx

    print('Nosify {} {}'.format(args.noise_group_num, args.group_noise_rate))
    for (s_idx, e_idx), g_noise_rate in zip(group_idx_lst, args.group_noise_rate):
        for i in range(s_idx, e_idx):
            for d_idx in dict_users[i]:
                train_dataset[d_idx][1] = noisify_label(train_dataset[d_idx][1], g_noise_rate)

    dataset_train = train_dataset
    
    # build model
    if args.model == 'cnn' and args.dataset == 'cifar':
        net_glob = CNNCifar(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'mnist':
        net_glob = CNNMnist(args=args).to(args.device)
    elif args.model == 'mlp':
        len_in = 1
        for x in img_size:
            len_in *= x
        net_glob = MLP(dim_in=len_in, dim_hidden=200, dim_out=args.num_classes).to(args.device)
    else:
        exit('Error: unrecognized model')
    print(net_glob)
    net_glob.train()

    # copy weights
    w_glob = net_glob.state_dict()

    # training
    loss_train = []
    cv_loss, cv_acc = [], []
    val_loss_pre, counter = 0, 0
    net_best = None
    best_loss = None
    val_acc_list, net_list = [], []
    
    
    # save results
    if args.save_dir is None:
        result_dir = './save/{}'.format(args.dataset)
    else:
        result_dir = './save/{}/{}/'.format(args.dataset, args.save_dir)
    
    result_f = 'fed_{}_{}_{}_C[{}]_IID[{}]_LR[{}]_MMT[{}]_NGN[{}]_GNR[{}].csv'.format(args.dataset, 
                                                                      args.model, 
                                                                      args.epochs, 
                                                                      args.frac, 
                                                                      args.iid,
                                                                      args.lr,
                                                                      args.momentum, 
                                                                      args.noise_group_num,
                                                                      args.group_noise_rate)
    
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
     
    f = open(result_dir + result_f,'w', newline='')
    wr = csv.writer(f)
    wr.writerow(['epoch','train_acc', 'train_loss', 'test_acc', 'test_loss'])

    if args.all_clients: 
        print("Aggregation over all clients")
        w_locals = [w_glob for i in range(args.num_users)]
    for iter in range(args.epochs):
        loss_locals = []
        if not args.all_clients:
            w_locals = []
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        
        if (iter + 1) in args.schedule:
            print("Learning Rate Decay Epoch {}".format(iter + 1))
            print("{} => {}".format(args.lr, args.lr * args.lr_decay))
            args.lr *= args.lr_decay
        
        for idx in idxs_users:
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
            w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))
            if args.all_clients:
                w_locals[idx] = copy.deepcopy(w)
            else:
                w_locals.append(copy.deepcopy(w))
            loss_locals.append(copy.deepcopy(loss))
        # update global weights
        w_glob = FedAvg(w_locals)

        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)

        # print result
        net_glob.eval()
        acc_train, loss_train = test_img(net_glob, dataset_train, args)
        acc_test, loss_test = test_img(net_glob, dataset_test, args)
        print('Round {:3d}'.format(iter))
        print("train acc: {}, train loss: {} \n test acc: {}, test loss: {}".format(acc_train.item(), 
                                                                                    loss_train, 
                                                                                    acc_test.item(), 
                                                                                    loss_test))
        
        wr.writerow([iter + 1, acc_train.item(), loss_train, acc_test.item(), loss_test])
        
    f.close()

