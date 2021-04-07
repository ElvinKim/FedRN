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
from models.Update import LocalUpdate, LocalUpdate_coteaching 
from models.Nets import MLP, CNNMnist, CNNCifar, CNN
from models.Fed import FedAvg
from models.test import test_img

from utils.cifar import CIFAR10, CIFAR100
from utils.mnist import MNIST

import csv
import os

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
    forget_rate = args.noise_rate
    exponent = 1
    num_gradual = args.num_gradual
    rate_schedule = np.ones(args.epochs)*forget_rate
    rate_schedule[:num_gradual] = np.linspace(0, forget_rate**exponent, num_gradual)
    
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
        if args.reproduce:
            trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), transforms.RandomCrop(32, padding=4), transforms.Resize(40)])
        else:
            trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
        dataset_train = CIFAR10(root='./data/cifar',
                                    download=True,  
                                    train=True, 
                                    transform=trans_cifar,
                                    noise_type=args.noise_type,
                                    noise_rate=args.noise_rate
                               )
        
        clean_dataset_train = CIFAR10(root='./data/cifar',
                                    download=True,  
                                    train=True, 
                                    transform=trans_cifar,
                                    noise_type="clean"
                               )
        
        dataset_test = CIFAR10(root='./data/cifar',
                                    download=True,  
                                    train=False, 
                                    transform=trans_cifar,
                                    noise_type=args.noise_type,
                                    noise_rate=args.noise_rate
                              )
        
        clean_dataset_test = CIFAR10(root='./data/cifar',
                                    download=True,  
                                    train=False, 
                                    transform=trans_cifar,
                                    noise_type="clean"
                              )
        
        if args.iid:
            dict_users = cifar_iid(clean_dataset_train, args.num_users)
        else:
            dict_users = cifar_noniid(clean_dataset_train, args.num_users)
    else:
        exit('Error: unrecognized dataset')
    img_size = dataset_train[0][0].shape

    # build model
    if args.model == 'cnn' and args.dataset == 'cifar':
        if args.reproduce:
            net_glob1 = CNN(input_channel=args.num_channels, n_outputs=args.num_classes).to(args.device)
            net_glob2 = CNN(input_channel=args.num_channels, n_outputs=args.num_classes).to(args.device)
        else:
            net_glob1 = CNNCifar(args=args).to(args.device)
            net_glob2 = CNNCifar(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'mnist':
        if args.reproduce:
            net_glob1 = CNN(input_channel=args.num_channels, n_outputs=args.num_classes).to(args.device)
            net_glob2 = CNN(input_channel=args.num_channels, n_outputs=args.num_classes).to(args.device)
        else:
            net_glob1 = CNNMnist(args=args).to(args.device)
            net_glob2 = CNNMnist(args=args).to(args.device)
    elif args.model == 'mlp':
        len_in = 1
        for x in img_size:
            len_in *= x
        net_glob1 = MLP(dim_in=len_in, dim_hidden=200, dim_out=args.num_classes).to(args.device)
        net_glob2 = MLP(dim_in=len_in, dim_hidden=200, dim_out=args.num_classes).to(args.device)
    else:
        exit('Error: unrecognized model')
    print(net_glob1)
    print(net_glob2)
    net_glob1.train()
    net_glob2.train()

    # copy weights
    w_glob1 = net_glob1.state_dict()
    w_glob2 = net_glob2.state_dict()

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
    
    result_f = 'fedCoteaching_{}_{}_{}_C[{}]_IID[{}]_LR[{}]_MMT[{}]_NT[{}]_NR[{}]_Tk[{}]_RePr[{}].csv'.format(args.dataset, 
                                                                                       args.model, 
                                                                                       args.epochs, 
                                                                                       args.frac, 
                                                                                       args.iid,
                                                                                       args.lr,
                                                                                       args.momentum,
                                                                                       args.noise_type,
                                                                                       args.noise_rate,
                                                                                       args.num_gradual,
                                                                                       args.reproduce)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
     
    f = open(result_dir + result_f,'w', newline='')
    wr = csv.writer(f)
    wr.writerow(['epoch','train_acc', 'train_loss', 'test_acc', 'test_loss'])
    
    if args.all_clients: 
        print("Aggregation over all clients")
        w_locals1 = [w_glob1 for i in range(args.num_users)]
        w_locals2 = [w_glob2 for i in range(args.num_users)]
        
    best_accuracy = 0
    last10_accuracies = []
    for iter in range(args.epochs):
        loss_locals1 = []
        loss_locals2 = []
        if not args.all_clients:
            w_locals1 = []
            w_locals2 = []
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)

        if (iter + 1) in args.schedule:
            print("Learning Rate Decay Epoch {}".format(iter + 1))
            print("{} => {}".format(args.lr, args.lr * args.lr_decay))
            args.lr *= args.lr_decay
            
        for idx in idxs_users:
            local = LocalUpdate_coteaching(args=args, dataset=dataset_train, idxs=dict_users[idx])
            w1, loss1, w2, loss2 = local.train_coteaching(net1=copy.deepcopy(net_glob1).to(args.device), net2=copy.deepcopy(net_glob2).to(args.device), rate_schedule=rate_schedule[iter])
            if args.all_clients:
                w_locals1[idx] = copy.deepcopy(w1)
                w_locals2[idx] = copy.deepcopy(w2)
            else:
                w_locals1.append(copy.deepcopy(w1))
                w_locals2.append(copy.deepcopy(w2))
            loss_locals1.append(copy.deepcopy(loss1))
            loss_locals2.append(copy.deepcopy(loss2))
        # update global weights
        w_glob1 = FedAvg(w_locals1)
        w_glob2 = FedAvg(w_locals2)

        # copy weight to net_glob
        net_glob1.load_state_dict(w_glob1)
        net_glob2.load_state_dict(w_glob2)

        # print result
        net_glob1.eval()
        net_glob2.eval()
        acc_train1, loss_train1 = test_img(net_glob1, dataset_train, args)
        acc_train2, loss_train2 = test_img(net_glob2, dataset_train, args)
        acc_test1, loss_test1 = test_img(net_glob1, dataset_test, args)
        acc_test2, loss_test2 = test_img(net_glob2, dataset_test, args)
        print('Round {:3d}'.format(iter))
        print("train acc1: {}, train loss1: {} \n train acc2: {}, train loss2: {} \n test acc1: {}, test loss1: {} \n test acc2: {}, test loss2: {}".format(acc_train1.item(), loss_train1, acc_train2.item(), loss_train2, acc_test1.item(), loss_test1, acc_test2.item(), loss_test2))
        
        wr.writerow([iter + 1, max(acc_train1.item(), acc_train2.item()), min(loss_train1, loss_train2), max(acc_test1.item(), acc_test2.item()), min(loss_test1, loss_test2)])
        
        if iter + 1 > args.epochs - 10:
            last10_accuracies.append(max(acc_test1.item(), acc_test2.item()))
        
        if best_accuracy < max(acc_test1.item(), acc_test2.item()):
            best_accuracy = max(acc_test1.item(), acc_test2.item())
            
    wr.writerow([])
    wr.writerow(['best_acc', 'last10_acc'])
    wr.writerow([best_accuracy, sum(last10_accuracies)/10])
    f.close()


