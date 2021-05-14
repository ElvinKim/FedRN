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

from utils.sampling import mnist_iid, mnist_noniid, cifar_iid, cifar_noniid, non_iid
from utils.options import args_parser
from models.Update import LocalUpdate, LocalUpdateLGFineTuning
from models.Nets import MLP, CNNMnist, CNNCifar, MobileNetCifar
from models.Fed import FedAvg
from models.test import test_img

from utils.cifar import CIFAR10, CIFAR100, CIFAR10Basic
from utils.mnist import MNIST

from utils.utils import noisify_label

import csv
import os
import random

if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    # for Jang's GPU usage
    torch.cuda.set_device(args.device)
    
    # Reproducing "Robust Fl with NLs"
    if args.reproduce:
        args.local_bs = 50
        args.lr = 0.15
        args.lr_decay = 1
        args.weight_decay = 0.0001

    # Seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(args.seed)
    
    
    # Co-teaching setup
    args.num_gradual = int(args.epochs * 0.2)
    forget_rate = args.forget_rate
    exponent = 1
    num_gradual = args.num_gradual
    rate_schedule = np.ones(args.epochs)*forget_rate
    rate_schedule[:num_gradual] = np.linspace(0, forget_rate**exponent, num_gradual)
    
    print(rate_schedule)
    
    # load dataset and split users
    if args.dataset == 'mnist':
        from six.moves import urllib    
        opener = urllib.request.build_opener()
        opener.addheaders = [('User-agent', 'Mozilla/5.0')]
        urllib.request.install_opener(opener)
        
        if args.reproduce:
            trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,)), transforms.RandomCrop(28, padding=4), transforms.Resize(32)])
        else: 
            trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        
        clearn_dataset_train = MNIST(root='./data/mnist',
                                download=True,  
                                train=True, 
                                transform=trans_mnist,
                                noise_type="clean",
                         )
        
        dataset_train = MNIST(root='./data/mnist',
                                download=True,  
                                train=True, 
                                transform=trans_mnist,
                                noise_type=args.noise_type,
                                noise_rate=args.noise_rate
                         )
        
        dataset_test = MNIST(root='./data/mnist',
                                   download=True,  
                                   train=False, 
                                   transform=transforms.ToTensor(),
                                   noise_type=args.noise_type,
                                   noise_rate=args.noise_rate
                            )
        
        clean_dataset_test = MNIST(root='./data/mnist',
                                   download=True,  
                                   train=False, 
                                   transform=transforms.ToTensor(),
                                   noise_type="clean"
                            )
        # sample users
        if args.iid:
            dict_users = mnist_iid(clearn_dataset_train, args.num_users)
        else:
            dict_users = mnist_noniid(clearn_dataset_train, args.num_users)
    elif args.dataset == 'cifar':
        trans_cifar10_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                               std=[0.229, 0.224, 0.225])])
        trans_cifar10_val = transforms.Compose([transforms.ToTensor(),
                                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                     std=[0.229, 0.224, 0.225])])
        dataset_train = CIFAR10Basic(root='./data/cifar',
                                    download=True,  
                                    train=True, 
                                    transform=trans_cifar10_train)
        
        dataset_test = CIFAR10Basic(root='./data/cifar',
                                    download=True,  
                                    train=False, 
                                    transform=trans_cifar10_val)
        
        if args.iid:
            dict_users = cifar_iid(dataset_train, args.num_users)
        else:
            if args.test_partition:
                dict_users, rand_set_all = non_iid(dataset_train, args.num_users, shard_per_user=2)
                dict_test_users, rand_set_all = non_iid(dataset_test, args.num_users, shard_per_user=2, rand_set_all=rand_set_all, mode='test')
            else:
                dict_users = cifar_noniid(dataset_train, args.num_users, partition=args.partition)
    else:
        exit('Error: unrecognized dataset')
    img_size = dataset_train[0][0].shape

    # noisify step
    if args.noise_type != "clean":
        if sum(args.noise_group_num) != args.num_users:
            exit('Error: sum of the number of noise group have to be equal the number of users')

        noise_rate_lst = []

        for noise_num, noise_rate in zip(args.noise_group_num, args.group_noise_rate):
            temp_lst = [noise_rate] * noise_num
            noise_rate_lst += temp_lst

        for i in range(args.num_users):
            d_idxs_lst = list(copy.deepcopy(dict_users[i]))
            noise_rate = noise_rate_lst[i]

            # for reproduction
            random.seed(args.seed)
            random.shuffle(d_idxs_lst)

            noise_index = int(len(d_idxs_lst) * noise_rate)

            for d_idx in d_idxs_lst[:noise_index]:
                true_label = dataset_train.train_labels[d_idx]
                update_label = noisify_label(true_label, num_classes=10, noise_type=args.noise_type)
                dataset_train.train_labels[d_idx] = update_label

    # build model
    if args.model == 'cnn' and args.dataset == 'cifar':
        if args.reproduce:
            net_glob = CNN(input_channel=args.num_channels, n_outputs=args.num_classes).to(args.device)
            net_local = CNN(input_channel=args.num_channels, n_outputs=args.num_classes).to(args.device)
        else:
            net_glob = CNNCifar(args=args).to(args.device)
            net_local = CNNCifar(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'mnist':
        if args.reproduce:
            net_glob = CNN(input_channel=args.num_channels, n_outputs=args.num_classes).to(args.device)
            net_local = CNN(input_channel=args.num_channels, n_outputs=args.num_classes).to(args.device)
        else:
            net_glob = CNNMnist(args=args).to(args.device)
            net_local = CNNMnist(args=args).to(args.device)
    elif args.model == 'mlp':
        len_in = 1
        for x in img_size:
            len_in *= x
        net_glob = MLP(dim_in=len_in, dim_hidden=200, dim_out=args.num_classes).to(args.device)
        net_local = MLP(dim_in=len_in, dim_hidden=200, dim_out=args.num_classes).to(args.device)
    elif args.model == "mobile":
        net_glob = MobileNetCifar().to(args.device)
        net_local = MobileNetCifar().to(args.device)
    else:
        exit('Error: unrecognized model')
        
    net_glob.train()

    # copy weights
    w_glob = net_glob.state_dict()
    
    # local weights
    w_local_lst = []
    
    for idx in range(args.num_users):
        w_local_lst.append(MobileNetCifar().to(args.device).state_dict())
    
    # training
    loss_train = []
    cv_loss, cv_acc = [], []
    val_loss_pre, counter = 0, 0
    net_best = None
    best_loss = None
    val_acc_list, net_list = [], []
   
    # save results
    if args.save_dir is None:
        result_dir = './save/'
    else:
        result_dir = './save/{}/'.format(args.save_dir)
    
    if args.iid:
        result_f = '{}_{}_{}_C[{}]_BS[{}]_LE[{}]_IID[{}]_LR[{}]_MMT[{}]_NT[{}]_NGN[{}]_GNR[{}]'.format(args.dataset, 
                                                                                                     args.model, 
                                                                                                     args.epochs, 
                                                                                                     args.frac, 
                                                                                                     args.local_bs, 
                                                                                                     args.local_ep, 
                                                                                                     args.iid,
                                                                                                     args.lr,
                                                                                                     args.momentum,
                                                                                                     args.noise_type,
                                                                                                     args.noise_group_num,
                                                                                                     args.group_noise_rate)
    else:
        result_f = '{}_{}_{}_C[{}]_BS[{}]_LE[{}]_IID[{}]_LR[{}]_MMT[{}]_NT[{}]_NGN[{}]_GNR[{}]_PT[{}]'.format(args.dataset, 
                                                                                                     args.model, 
                                                                                                     args.epochs, 
                                                                                                     args.frac, 
                                                                                                     args.local_bs, 
                                                                                                     args.local_ep, 
                                                                                                     args.iid,
                                                                                                     args.lr,
                                                                                                     args.momentum,
                                                                                                     args.noise_type,
                                                                                                     args.noise_group_num,
                                                                                                     args.group_noise_rate, 
                                                                                                     args.partition)

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
     
    f = open(result_dir + result_f + ".csv", 'w', newline='')
    wr = csv.writer(f)
    wr.writerow(['epoch','train_acc', 'train_loss', 'test_acc', 'test_loss'])
    
    # Option Save
    with open(result_dir + result_f + ".txt", 'w') as option_f:
        option_f.write(str(args)) 
    
    if args.all_clients: 
        print("Aggregation over all clients")
        w_locals = [w_glob1 for i in range(args.num_users)]
        
    best_accuracy = 0
    last10_accuracies = []
    
    for iter in range(args.epochs):
        if not args.all_clients:
            w_glob_lst = []
            
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
            
        for idx in idxs_users:
            local = LocalUpdateLGFineTuning(args=args, dataset=dataset_train, idxs=dict_users[idx])
            net_local.load_state_dict(w_local_lst[idx])
            
            if args.test_partition:
                test_dataset = []
                for d_idx in dict_test_users[idx]:
                    test_dataset.append(dataset_test[d_idx])
                
                acc_local, loss_local = test_img(net_local, test_dataset, args)
                print("before tuning", idx, acc_local)
            
            w_g, w_l, loss = local.train(g_net=copy.deepcopy(net_glob).to(args.device),
                                         l_net=copy.deepcopy(net_local).to(args.device),
                                         forget_rate=rate_schedule[iter])
            net_local.load_state_dict(w_l)
            
            if args.test_partition:
                acc_local, loss_local = test_img(net_local, test_dataset, args)
                print("after tuning", idx, acc_local)
                
            # w_local_lst[idx] = copy.deepcopy(w_l)
            w_glob_lst.append(copy.deepcopy(w_g)) 
            
        # update global weights
        w_glob = FedAvg(w_glob_lst)

        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)

        # print result
        net_glob.eval()
        
        acc_train, loss_train = test_img(net_glob, dataset_train, args)
        acc_test, loss_test = test_img(net_glob, dataset_test, args)
        
        print('Round {:3d}'.format(iter))
        print("train acc: {}, train loss: {} \n test acc: {}, test loss: {} ".format(acc_train.item(), 
                                                                                     loss_train, 
                                                                                     acc_test.item(), 
                                                                                     loss_test))
        
        wr.writerow([iter + 1, acc_train.item(), loss_train, acc_test.item(), loss_test])
        
    f.close()


