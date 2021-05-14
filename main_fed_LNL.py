#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import csv
import copy
import numpy as np
import os
import random

import torchvision
from torchvision import datasets, transforms
import torch

from utils import CIFAR10Basic, MNIST, Logger
from utils.sampling import sample_iid, sample_noniid
from utils.options import args_parser
from utils.utils import noisify_label

from models.Update import get_local_update_objects
from models.Nets import get_model
from models.Fed import FedAvg
from models.test import test_img

try:
    import nsml
    USE_NSML = True
except:
    USE_NSML = False


def init_local_weights(all_clients, w_glob, num_users):
    w_locals = []

    if all_clients:
        print("Aggregation over all clients")
        w_locals = [w_glob for i in range(num_users)]

    return w_locals


if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    args.schedule = [int(x) for x in args.schedule]
    args.warm_up = int(args.epochs * 0.2) \
        if args.method in ['dividemix', 'gmix'] \
        else 0
    use_2_models = args.method in ['coteaching', 'coteaching+', 'dividemix']
    print(use_2_models)

    for x in vars(args).items():
        print(x)

    print(torch.__version__)
    print(torchvision.__version__)

    # Seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(args.seed)

    ##############################
    # Load dataset and split users
    ##############################
    if args.dataset == 'mnist':
        from six.moves import urllib

        opener = urllib.request.build_opener()
        opener.addheaders = [('User-agent', 'Mozilla/5.0')]
        urllib.request.install_opener(opener)

        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

        dataset_args = dict(
            root='./data/mnist',
            download=True,
        )
        dataset_train = MNIST(
            train=True,
            transform=trans_mnist,
            noise_type="clean",
            **dataset_args,
        )
        dataset_test = MNIST(
            train=False,
            transform=transforms.ToTensor(),
            noise_type="clean",
            **dataset_args,
        )
        num_classes = 10

    elif args.dataset == 'cifar':
        trans_cifar10_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])],
        )
        trans_cifar10_val = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])],
        )
        dataset_train = CIFAR10Basic(
            root='./data/cifar',
            download=True,
            train=True,
            transform=trans_cifar10_train,
        )
        dataset_test = CIFAR10Basic(
            root='./data/cifar',
            download=True,
            train=False,
            transform=trans_cifar10_val,
        )
        num_classes = 10

    else:
        raise NotImplementedError('Error: unrecognized dataset')

    labels = np.array(dataset_train.train_labels)
    num_imgs = len(dataset_train) // args.num_shards
    args.img_size = dataset_train[0][0].shape  # used to get model
    args.num_classes = num_classes

    # Sample users (iid / non-iid)
    if args.iid:
        dict_users = sample_iid(dataset_train, args.num_users)
    else:
        dict_users = sample_noniid(
            labels=labels,
            num_users=args.num_users,
            num_shards=args.num_shards,
            num_imgs=num_imgs,
        )
    
    ##############################
    # Add label noise to data
    ##############################
    if args.noise_type != "clean" and args.group_noise_rate:
        if sum(args.noise_group_num) != args.num_users:
            exit('Error: sum of the number of noise group have to be equal the number of users')

        # noise rate for each user
        user_noise_rates = []
        for num_users_in_group, group_noise_rate in zip(args.noise_group_num, args.group_noise_rate):
            user_noise_rates += [group_noise_rate] * num_users_in_group

        for i in range(args.num_users):
            data_indices = list(copy.deepcopy(dict_users[i]))
            noise_rate = user_noise_rates[i]

            # for reproduction
            random.seed(args.seed)
            random.shuffle(data_indices)

            noise_index = int(len(data_indices) * noise_rate)

            for d_idx in data_indices[:noise_index]:
                true_label = dataset_train.train_labels[d_idx]
                noisy_label = noisify_label(true_label, num_classes=num_classes, noise_type=args.noise_type)
                dataset_train.train_labels[d_idx] = noisy_label
    else:
        user_noise_rates = [0] * args.num_users
    print(user_noise_rates)

    # for logging purposes
    log_train_data_loader = torch.utils.data.DataLoader(dataset_train, batch_size=args.bs)
    log_test_data_loader = torch.utils.data.DataLoader(dataset_test, batch_size=args.bs)

    ##############################
    # Build model
    ##############################
    net_glob = get_model(args)
    net_glob = net_glob.to(args.device)
    print(net_glob)
    w_glob = net_glob.state_dict()  # copy weights

    if use_2_models:
        net_glob2 = get_model(args)
        net_glob2 = net_glob2.to(args.device)
        w_glob2 = net_glob2.state_dict()  # copy weights

    ##############################
    # Training
    ##############################
    logger = Logger(args, use_2_models)

    all_loss_lst = [[[], []] for _ in range(args.num_users)]

    forget_rate_schedule = []
    if args.method in ['coteaching', 'coteaching+', 'gfilter']:
        forget_rate = args.forget_rate
        exponent = 1
        num_gradual = int(args.epochs * 0.2)
        forget_rate_schedule = np.ones(args.epochs) * forget_rate
        forget_rate_schedule[:num_gradual] = np.linspace(0, forget_rate ** exponent, num_gradual)

    # Initialize local weights
    w_locals = init_local_weights(all_clients=args.all_clients, w_glob=w_glob, num_users=args.num_users)
    if use_2_models:
        w_locals2 = init_local_weights(all_clients=args.all_clients, w_glob=w_glob2, num_users=args.num_users)

    # Initialize local update objects
    local_update_objects = get_local_update_objects(args, dataset_train, dict_users, user_noise_rates)

    for epoch in range(args.epochs):
        if (epoch + 1) in args.schedule:
            print("Learning Rate Decay Epoch {}".format(epoch + 1))
            print("{} => {}".format(args.lr, args.lr * args.lr_decay))
            args.lr *= args.lr_decay

        if len(forget_rate_schedule) > 0:
            args.forget_rate = forget_rate_schedule[epoch]

        loss_locals = []
        loss_locals2 = []
        # Reset local weights if necessary
        if not args.all_clients:
            w_locals = []
            w_locals2 = []

        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        
        # Local Update
        for idx in [31, 31]:
            local = local_update_objects[idx]
            local.args = args

            if use_2_models:
                if not args.warm_up or epoch <= args.warm_up:
                    w, loss, w2, loss2 = local.train(net=copy.deepcopy(net_glob).to(args.device),
                                                     net2=copy.deepcopy(net_glob2).to(args.device))
                else:
                    w1, w2, all_loss = local.train_2_phase(g_epoch=epoch,
                                                           warmup=args.warm_up,
                                                           net=copy.deepcopy(net_glob).to(args.device),
                                                           net2=copy.deepcopy(net_glob2).to(args.device),
                                                           all_loss=all_loss_lst[idx])
                    all_loss_lst[idx] = all_loss

                # Second model
                if args.all_clients:
                    w_locals2[idx] = copy.deepcopy(w2)
                else:
                    w_locals2.append(copy.deepcopy(w2))
                loss_locals2.append(copy.deepcopy(loss2))

            else:
                if not args.warm_up or epoch <= args.warm_up:
                    # Local weights, losses
                    w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))
                else:
                    w, loss = local.train_2_phase(net=copy.deepcopy(net_glob).to(args.device),
                                                  g_epoch=epoch)

            # Single / First model
            if args.all_clients:
                w_locals[idx] = copy.deepcopy(w)
            else:
                w_locals.append(copy.deepcopy(w))
            loss_locals.append(copy.deepcopy(loss))
            print("-" * 20)
        import sys; sys.exit()

        w_glob = FedAvg(w_locals)  # update global weights
        net_glob.load_state_dict(w_glob)  # copy weight to net_glob
        train_acc, train_loss = test_img(net_glob, log_train_data_loader, args)
        test_acc, test_loss = test_img(net_glob, log_test_data_loader, args)

        results = dict(
            train_acc=train_acc,
            train_loss=train_loss,
            test_acc=test_acc,
            test_loss=test_loss,
        )

        if use_2_models:
            w_glob2 = FedAvg(w_locals2)
            net_glob2.load_state_dict(w_glob2)
            train_acc2, train_loss2 = test_img(net_glob2, log_train_data_loader, args)
            test_acc2, test_loss2 = test_img(net_glob2, log_test_data_loader, args)

            results2 = dict(
                train_acc2=train_acc2,
                train_loss2=train_loss2,
                test_acc2=test_acc2,
                test_loss2=test_loss2,
            )
            results = {**results, **results2}

        print('Round {:3d}'.format(epoch))
        print(' - '.join([f'{k}: {v:.6f}' for k, v in results.items()]))

        logger.write(epoch=epoch + 1, **results)

        if USE_NSML and nsml.IS_ON_NSML:
            nsml_results = {result_key.replace('_', '__'): result_value
                            for result_key, result_value in results.items()}
            nsml.report(
                summary=True,
                step=epoch,
                epoch=epoch,
                lr=args.lr,
                **nsml_results,
            )

    logger.close()
