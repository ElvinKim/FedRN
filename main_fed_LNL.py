#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import copy
import numpy as np
import random
from collections import Counter

import torchvision
import torch
from torch.utils.data import DataLoader

from utils import Logger, NoiseLogger, load_dataset
from utils.sampling import sample_iid, sample_noniid_shard, sample_dirichlet, valid_sampling
from utils.options import args_parser
from utils.utils import noisify_label

from models.Update import get_local_update_objects, DatasetSplit
from models.Nets import get_model
from models.Fed import LocalModelWeights
from models.test import test_img
import nsml
import time
import os
import csv


if __name__ == '__main__':
    start = time.time()
    # parse args
    args = args_parser()
    args.device = torch.device(
        'cuda:{}'.format(args.gpu)
        if torch.cuda.is_available() and args.gpu != -1
        else 'cpu',
    )
    args.schedule = [int(x) for x in args.schedule]
    args.send_2_models = args.method in ['coteaching', 'coteaching+', 'dividemix', ]

    for x in vars(args).items():
        print(x)

    if not torch.cuda.is_available():
        exit('ERROR: Cuda is not available!')
    print('torch version: ', torch.__version__)
    print('torchvision version: ', torchvision.__version__)

    # Seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(args.seed)

    # Arbitrary gaussian noise
    gaussian_noise = torch.randn(1, 3, 32, 32)

    ##############################
    # Load dataset and split users
    ##############################
    dataset_train, dataset_test, imagenet_val, args.num_classes = load_dataset(args.dataset)
    labels = np.array(dataset_train.train_labels)
    img_size = dataset_train[0][0].shape  # used to get model
    args.img_size = int(img_size[1])
    print(img_size)
    
    # Sample users (iid / non-iid)
    if args.iid:
        print('iid')
        dict_users = sample_iid(labels, args.num_users)
    elif args.partition == 'shard':
        print(args.partition)
        dict_users = sample_noniid_shard(
            labels=labels,
            num_users=args.num_users,
            num_shards=args.num_shards,
        )
  
    elif args.partition == 'dirichlet':
        print(args.partition)
        dict_users = sample_dirichlet(
            labels=labels,
            num_users=args.num_users,
            alpha=args.dd_alpha,
        )
 
    tmp_true_labels = list(copy.deepcopy(dataset_train.train_labels))
    tmp_true_labels = torch.tensor(tmp_true_labels).to(args.device)


    valid_dict_users = valid_sampling(dict_users, args.num_users, dataset_test, tmp_true_labels)
    
    ##############################
    # Add label noise to data
    ##############################
    if sum(args.noise_group_num) != args.num_users:
        exit('Error: sum of the number of noise group have to be equal the number of users')

    if len(args.group_noise_rate) == 1:
        args.group_noise_rate = args.group_noise_rate * 2

    if not len(args.noise_group_num) == len(args.group_noise_rate) and \
            len(args.group_noise_rate) * 2 == len(args.noise_type_lst):
        exit('Error: The noise input is invalid.')

    args.group_noise_rate = [(args.group_noise_rate[i * 2], args.group_noise_rate[i * 2 + 1])
                             for i in range(len(args.group_noise_rate) // 2)]

    user_noise_type_rates = []
    for num_users_in_group, noise_type, (min_group_noise_rate, max_group_noise_rate) in zip(
            args.noise_group_num, args.noise_type_lst, args.group_noise_rate):
        noise_types = [noise_type] * num_users_in_group

        step = (max_group_noise_rate - min_group_noise_rate) / num_users_in_group
        noise_rates = np.array(range(num_users_in_group)) * step + min_group_noise_rate

        user_noise_type_rates += [*zip(noise_types, noise_rates)]

    user_noisy_data = {user: [] for user in dict_users}
    for user, (user_noise_type, user_noise_rate) in enumerate(user_noise_type_rates):
        if user_noise_type != "clean":
            data_indices = list(copy.deepcopy(dict_users[user]))

            # for reproduction
            random.seed(args.seed)
            random.shuffle(data_indices)

            noise_index = int(len(data_indices) * user_noise_rate)

            for d_idx in data_indices[:noise_index]:
                true_label = dataset_train.train_labels[d_idx]
                noisy_label = noisify_label(true_label, num_classes=args.num_classes, noise_type=user_noise_type)
                dataset_train.train_labels[d_idx] = noisy_label
                user_noisy_data[user].append(d_idx)
        else:
            if user_noise_rate != 0:
                data_indices = list(copy.deepcopy(dict_users[user]))

                # for reproduction
                random.seed(args.seed)
                random.shuffle(data_indices)

                delete_index = int(len(data_indices) * user_noise_rate)

                dict_users[user] = data_indices[delete_index:]

    total_noise_cnt = 0
    for user, user_noise_type_rate in enumerate(user_noise_type_rates):
        print("USER {} - {} - {}".format(user, user_noise_type_rate,
                                         int(len(dict_users[user]) * user_noise_type_rates[user][1])))
        total_noise_cnt += int(len(dict_users[user]) * user_noise_type_rates[user][1])

    print("Global Noise Rate : {}".format(total_noise_cnt / 50000))

    # for logging purposes
    logging_args = dict(
        batch_size=args.bs,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    log_train_data_loader = torch.utils.data.DataLoader(dataset_train, **logging_args)
    log_test_data_loader = torch.utils.data.DataLoader(dataset_test, **logging_args)
    if imagenet_val is not None:
        log_imagenet_val_data_loader = torch.utils.data.DataLoader(imagenet_val, **logging_args)

    ##############################
    # Build model
    ##############################
    net_glob = get_model(args)
    net_glob = net_glob.to(args.device)
    print(net_glob)
    
    if args.send_2_models:
        net_glob2 = get_model(args)
        net_glob2 = net_glob2.to(args.device)

    ##############################
    # Training
    ##############################
    logger = Logger(args, args.send_2_models)
    noise_logger = NoiseLogger(args, user_noisy_data, dict_users)

    forget_rate_schedule = []

    if args.method in ['coteaching', 'coteaching+']:
        if args.forget_rate_schedule == "fix":
            num_gradual = args.warmup_epochs

        elif args.forget_rate_schedule == 'stairstep':
            num_gradual = args.epochs

        else:
            exit("Error: Forget rate schedule - fix or stairstep")

        forget_rate = args.forget_rate
        exponent = 1
        forget_rate_schedule = np.ones(args.epochs) * forget_rate
        forget_rate_schedule[:num_gradual] = np.linspace(0, forget_rate ** exponent, num_gradual)

    print("Forget Rate Schedule")
    print(forget_rate_schedule)

    pred_user_noise_rates = [args.forget_rate] * args.num_users

    # Initialize local model weights
    fed_args = dict(
        all_clients=args.all_clients,
        num_users=args.num_users,
        method=args.fed_method,
        dict_users=dict_users,
    )

    local_weights = LocalModelWeights(net_glob=net_glob, **fed_args)
    if args.send_2_models:
        local_weights2 = LocalModelWeights(net_glob=net_glob2, **fed_args)

    f_G = torch.randn(args.num_classes, args.feature_dim, device=args.device)

    # Initialize local update objects
    local_update_objects = get_local_update_objects(
        args=args,
        dataset_train=dataset_train,
        dict_users=dict_users,
        noise_rates=pred_user_noise_rates,
        net_glob=net_glob,
        noise_logger=noise_logger,
        gaussian_noise=gaussian_noise,
        user_noisy_data=user_noisy_data
    )

    cos_sim = torch.nn.CosineSimilarity()
    
    for i in range(args.num_users):
        local = local_update_objects[i]
        local.weight = copy.deepcopy(net_glob.state_dict())
        
        
    # for val. acc. check    
    if args.save_dir is None:
        result_dir = './save/'
    else:
        result_dir = './save/{}/'.format(args.save_dir)
            
    result_f = 'val_acc_summary_nei[{}]_alpha[{}]_{}_{}_{}_{}_NT[{}]_GNR[{}]_PT[{}]_SHARDNUM[{}]'.format(
                args.num_neighbors,
                args.w_alpha,
                args.dataset,
                args.method,
                args.model,
                args.epochs,
                args.noise_type_lst,
                args.group_noise_rate,
                args.partition,
                args.num_shards
                )

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    
    f = open(result_dir + result_f + ".csv", 'w', newline='')
    wr = csv.writer(f)
    wr.writerow(['epoch', 'user_id', 'val_acc'])


    for epoch in range(args.epochs):
        if (epoch + 1) in args.schedule:
            print("Learning Rate Decay Epoch {}".format(epoch + 1))
            print("{} => {}".format(args.lr, args.lr * args.lr_decay))
            args.lr *= args.lr_decay

        if len(forget_rate_schedule) > 0:
            args.forget_rate = forget_rate_schedule[epoch]

        local_losses = []
        local_losses2 = []
        args.g_epoch = epoch
        f_locals = []

        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)

        # Local Update
        for client_num, idx in enumerate(idxs_users):
            local = local_update_objects[idx]
            local.args = args
            # local.args.local_bs = int(len(dict_users[idx]) / 10)

            if args.send_2_models:
                w, loss, w2, loss2 = local.train(
                    client_num,
                    copy.deepcopy(net_glob).to(args.device),
                    copy.deepcopy(net_glob2).to(args.device),
                )
                local_weights.update(idx, w)
                local_losses.append(copy.deepcopy(loss))
                local_weights2.update(idx, w2)
                local_losses2.append(copy.deepcopy(loss2))

            else:
                if args.method == "fedrn":
                    
                    if epoch < args.warmup_epochs:
                        w, loss = local.train_phase1(client_num, copy.deepcopy(net_glob).to(args.device))
                    else:
                        score_list = []
                        sim_list = [cos_sim(local_update_objects[idx].arbitrary_output.to(args.device),
                                            local_update_objects[n_i].arbitrary_output.to(args.device)) for n_i in
                                    range(args.num_users)]
                        exp_list = [local_update_objects[n_i].expertise for n_i in range(args.num_users)]

                        # Similarity & expertise normalizing
                        sim_list = [((i-min(sim_list))/(max(sim_list)-min(sim_list))).item() for i in sim_list]
                        exp_list = [(i-min(exp_list))/(max(exp_list)-min(exp_list)) for i in exp_list]
                        
                        for index, (e, s) in enumerate(zip(exp_list, sim_list)):
                            if index != idx:
                                score = args.w_alpha * e + (1 - args.w_alpha) * s
                                score_list.append([score, index])

                        score_list.sort(key=lambda x: x[0], reverse=True)
                        
                        prev_score = args.w_alpha * exp_list[idx] + (1 - args.w_alpha)
                        
                        neighbor_lst = []
                        neighbor_score_lst = []
                        print('--------------------------------------------------------------')
 
                        print('[Index & Score Summary]')
                        print('Client          - idx: {}, sim: {:.5f}, exp: {:.5f}'.format(idx, 1, exp_list[idx]))
                        data_counter = Counter(tmp_true_labels[dict_users[idx]].tolist())
                        print('Client Data     - {}'.format(sorted(data_counter.items())))
                        
                        for n_index in range(args.num_neighbors):
                            if args.random_neighbor:
                                rand_list =list(range(0,100))
                                rand_list.remove(idx)
                                neighbor_idx = random.choice(rand_list)
                                neighbor_score_lst.append(prev_score)
                            else:    
                                neighbor_idx = score_list[n_index][1]
                                neighbor_score_lst.append(score_list[n_index][0])
                                
                            neighbor_net = copy.deepcopy(local_update_objects[neighbor_idx].net1)
                            neighbor_lst.append(neighbor_net)

                            print('Neighbor {}      - idx: {}, sim: {:.5f}, exp: {:.5f}'.format(n_index+1, neighbor_idx, sim_list[neighbor_idx], exp_list[neighbor_idx]))
                            data_counter = Counter(tmp_true_labels[dict_users[neighbor_idx]].tolist())
                            print('Neighbor {} Data - {}'.format(n_index+1, sorted(data_counter.items())))
                        
                        
                        w, loss = local.train_phase2(client_num, 
                                                     copy.deepcopy(net_glob).to(args.device),
                                                     prev_score,
                                                     neighbor_lst,
                                                     neighbor_score_lst)
                    

                else:
                    w, loss = local.train(client_num, copy.deepcopy(net_glob).to(args.device))

                # local_weights.update(idx, w, len(dict_users[idx]))
                local_weights.update(idx, w)
                local_losses.append(copy.deepcopy(loss))

        w_glob = local_weights.average()  # update global weights
        net_glob.load_state_dict(w_glob, strict=False)  # copy weight to net_glob

        local_weights.init()

        train_acc, train_loss = test_img(net_glob, log_train_data_loader, args)
        test_acc, test_loss = test_img(net_glob, log_test_data_loader, args)

        # for logging purposes
        results = dict(train_acc=train_acc, train_loss=train_loss,
                       test_acc=test_acc, test_loss=test_loss, )

        if imagenet_val is not None:
            test_imagenet_acc, test_imagenet_loss = test_img(net_glob, log_imagenet_val_data_loader, args)
            results['test_imagenetacc'] = test_imagenet_acc
            results['test_imagenetloss'] = test_imagenet_loss

        if args.send_2_models:
            w_glob2 = local_weights2.average()
            net_glob2.load_state_dict(w_glob2)
            local_weights2.init()
            # for logging purposes
            train_acc2, train_loss2 = test_img(net_glob2, log_train_data_loader, args)
            test_acc2, test_loss2 = test_img(net_glob2, log_test_data_loader, args)
            results2 = dict(train_acc2=train_acc2, train_loss2=train_loss2,
                            test_acc2=test_acc2, test_loss2=test_loss2, )
            if imagenet_val is not None:
                test_imagenet_acc2, test_imagenet_loss2 = test_img(net_glob, log_imagenet_val_data_loader, args)
                results2['test_imagenetacc2'] = test_imagenet_acc2
                results2['test_imagenetloss2'] = test_imagenet_loss2

            results = {**results, **results2}

        logger.write(epoch=epoch + 1, **results)

        if nsml.IS_ON_NSML:
            nsml_results = {result_key.replace('_', '__'): result_value
                            for result_key, result_value in results.items()}
            nsml.report(
                summary=True,
                step=epoch,
                epoch=epoch,
                lr=args.lr,
                **nsml_results,
            )

    f.close()
    logger.close()
    noise_logger.close()
    print("time :", time.time() - start)
