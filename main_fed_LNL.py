#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import csv
import copy
import numpy as np
import os
import random

from torchvision import datasets, transforms
import torch

from utils.cifar import CIFAR10, CIFAR100, CIFAR10Basic
from utils.mnist import MNIST
from utils.sampling import sample_iid, sample_noniid
from utils.options import args_parser
from utils.utils import noisify_label

from models.Update import get_local_update_objects
from models.Nets import get_model
from models.Fed import FedAvg
from models.test import test_img
import nsml


if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    args.schedule = [int(x) for x in args.schedule]
    for x in vars(args).items():
        print(x)

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

        # noisy_dataset_train = MNIST(
        #     train=True,
        #     transform=trans_mnist,
        #     noise_type=args.noise_type,
        #     noise_rate=args.noise_rate,
        #     **dataset_args,
        # )
        # noisy_dataset_test = MNIST(
        #     train=False,
        #     transform=transforms.ToTensor(),
        #     noise_type=args.noise_type,
        #     noise_rate=args.noise_rate,
        #     **dataset_args,
        # )

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
    args.img_size = dataset_train[0][0].shape    # used to get model
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

    ##############################
    # Build model
    ##############################
    net_glob = get_model(args)
    net_glob = net_glob.to(args.device)
    net_glob.train()
    print(net_glob)

    # copy weights
    w_glob = net_glob.state_dict()

    ##############################
    # Training
    ##############################
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
        result_f = 'fedLNL_{}_{}_{}_{}_C[{}]_BS[{}]_LE[{}]_IID[{}]_LR[{}]_MMT[{}]_NT[{}]_NGN[{}]_GNR[{}]_PT[{}]'.format(
            args.dataset,
            args.method,
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
    else:
        result_f = 'fedLNL_{}_{}_{}_{}_C[{}]_BS[{}]_LE[{}]_IID[{}]_LR[{}]_MMT[{}]_NT[{}]_NGN[{}]_GNR[{}]'.format(
            args.dataset,
            args.method,
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

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    f = open(result_dir + result_f + ".csv", 'w', newline='')
    wr = csv.writer(f)
    wr.writerow(['epoch', 'train_acc', 'train_loss', 'test_acc', 'test_loss'])

    # Option Save
    with open(result_dir + result_f + ".txt", 'w') as option_f:
        option_f.write(str(args))

    if args.all_clients:
        print("Aggregation over all clients")
        w_locals = [w_glob for i in range(args.num_users)]

    local_update_objects = get_local_update_objects(args, dataset_train, dict_users, user_noise_rates)

    # for logging purposes
    log_train_data_loader = torch.utils.data.DataLoader(dataset_train, batch_size=args.bs)
    log_test_data_loader = torch.utils.data.DataLoader(dataset_test, batch_size=args.bs)

    for epoch in range(args.epochs):
        if (epoch + 1) in args.schedule:
            print("Learning Rate Decay Epoch {}".format(epoch + 1))
            print("{} => {}".format(args.lr, args.lr * args.lr_decay))
            args.lr *= args.lr_decay

        loss_locals = []
        # Reset local weights if necessary
        if not args.all_clients:
            w_locals = []

        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)

        # Local Update
        for idx in idxs_users:
            local = local_update_objects[idx]
            local.args = args

            # Local weights, losses
            w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))
            if args.all_clients:
                w_locals[idx] = copy.deepcopy(w)
            else:
                w_locals.append(copy.deepcopy(w))
            loss_locals.append(copy.deepcopy(loss))

        # Update global weights
        w_glob = FedAvg(w_locals)

        # Copy weight to net_glob
        net_glob.load_state_dict(w_glob)

        # Print results
        net_glob.eval()
        acc_train, loss_train = test_img(net_glob, log_train_data_loader, args)
        acc_test, loss_test = test_img(net_glob, log_test_data_loader, args)

        if nsml.IS_ON_NSML:
            nsml.report(
                summary=True,
                step=epoch,
                epoch=epoch,
                lr=args.lr,
                train__acc=acc_train,
                train__loss=loss_train,
                test__acc=acc_test,
                test__loss=loss_test,
            )

        print('Round {:3d}'.format(epoch))
        print("train acc: {}, train loss: {:.6f} \ntest acc: {}, test loss: {:.6f}".format(
            acc_train,
            loss_train,
            acc_test,
            loss_test,
        ))
        wr.writerow([epoch + 1, acc_train, loss_train, acc_test, loss_test])

    f.close()
