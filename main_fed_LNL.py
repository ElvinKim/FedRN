#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import copy
import numpy as np
import random

import torchvision
import torch
from torch.utils.data import DataLoader

from utils import Logger, NoiseLogger, load_dataset
from utils.sampling import sample_iid, sample_noniid_shard, sample_dirichlet
from utils.options import args_parser
from utils.utils import noisify_label
from utils.logger import get_loss_dist

from models.Update import get_local_update_objects
from models.Nets import get_model
from models.Fed import LocalModelWeights
from models.test import test_img
import nsml
import time
# from pyemd import emd


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

    # Reproducing "Robust Fl with NLs"
    if args.reproduce and args.method == 'RFL':
        args.weight_decay = 0.0001
        args.lr = 0.25
        args.model = 'cnn9'
        args.feature_return = True

    # determine feature dimension
    if args.model == 'cnn9':
        args.feature_dim = 128
    elif args.model == 'cnn4conv':
        args.feature_dim = 256
    else:
        # Otherwise, need to check
        args.feature_dim = 0

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
    args.img_size = dataset_train[0][0].shape  # used to get model

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
    # print(dict_users)

    tmp_true_labels = list(copy.deepcopy(dataset_train.train_labels))
    tmp_true_labels = torch.tensor(tmp_true_labels).to(args.device)

    '''
    ##############################
    # Get Smart Neighbors
    ##############################
    
    # Calculate EMD
    user_label_dict = {}

    for i in range(args.num_users):
        user_label_dict[i] = [0] * args.num_classes

        for d in dict_users[i]:
            user_label_dict[i][dataset_train[d][1]] += 1

        user_data_cnt = len(dict_users[i])
        user_label_dict[i] = np.array(user_label_dict[i]) / user_data_cnt

    emd_matrix = pd.DataFrame(np.zeros((args.num_users, args.num_users)))

    for i in range(args.num_users):
        for j in range(args.num_users):
            emd_matrix[i][j] = emd(user_label_dict[i], user_label_dict[j], 1-np.identity(10))

    # Get Optimal Smart Neighbors
    if args.group_noise_rate[0] == 0.4:
        neighbor_cnt_threshold = 100
    else:
        neighbor_cnt_threshold = 50

    neighbor_dict = {}

    for i in range(args.num_users):
        n_lst = []

        top_idx_lst = emd_matrix[i].nsmallest(n=50).index.tolist()
        top_idx_lst.remove(i)

        for n_i in top_idx_lst:
            if n_i > neighbor_cnt_threshold:
                continue
            if len(n_lst) == args.num_neighbors:
                break

            n_lst.append(n_i)

        print(i, user_label_dict[i])
        for n_i in n_lst:
            print(n_i, user_label_dict[n_i])

        print("-" * 20)

        neighbor_dict[i] = n_lst
    '''
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

    # group_noise_rate = [(min noise rate, max noise rate)]
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

    ##############################
    # Build model
    ##############################
    net_glob = get_model(args)
    net_glob = net_glob.to(args.device)
    print(net_glob)

    if args.method in ['global_with_neighbors']:
        temp_net = copy.deepcopy(net_glob)

    if args.send_2_models:
        net_glob2 = get_model(args)
        net_glob2 = net_glob2.to(args.device)

    net_local_lst = None
    if args.method in ['lgfinetune', 'lgteaching', 'lgcorrection']:
        net_local_lst = []
        for i in range(args.num_users):
            net_local_lst.append(net_glob.to(args.device))

    ##############################
    # Training
    ##############################
    logger = Logger(args, args.send_2_models)
    noise_logger = NoiseLogger(args, user_noisy_data, dict_users)

    forget_rate_schedule = []

    if args.method in ['coteaching', 'coteaching+', 'finetune', 'RFL', "global_model"]:
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
        net_local_lst=net_local_lst,
        noise_logger=noise_logger,
        tmp_true_labels=tmp_true_labels,
        gaussian_noise=gaussian_noise,
    )

    expertise_list = [0 for i in range(args.num_users)]
    inference_list = [torch.zeros(1, args.num_classes) for i in range(args.num_users)]
    sim = torch.nn.CosineSimilarity()

    for i in range(args.num_users):
        local = local_update_objects[i]
        local.weight = copy.deepcopy(net_glob.state_dict())

    #     if args.method in ['global_with_neighbors']:
    #         if "cifar10" in args.dataset :
    #             arbitrary_input = torch.randn((1, 3, 32, 32)).to(args.device)
    #         elif "mnist" in args.dataset:
    #             arbitrary_input = torch.randn((1, 1, 28, 28)).to(args.device)

    #         arbitrary_output_lst = [None for _ in range(args.num_users)]

    #         neighbor_dict = {i: [] for i in range(args.num_users)}
    #         sim_dict = {i: [] for i in range(args.num_users)}

    #         select_check_lst = [i for i in range(args.num_users)]

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

        # total_data_cnt = sum([len(dict_users[idx]) for idx in idxs_users])

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
                if args.method == 'RFL':
                    w, loss, f_k = local.train(copy.deepcopy(net_glob).to(args.device),
                                               copy.deepcopy(f_G).to(args.device), client_num)
                    f_locals.append(f_k)
                elif args.method == "global_with_neighbors":
                    neighbor_local = None

                    if epoch > args.warmup_epochs:
                        neighbor_local = []

                        for n_idx in neighbor_dict[idx]:
                            temp_net.load_state_dict(local_update_objects[n_idx].weight)
                            neighbor_local.append(copy.deepcopy(temp_net))

                    w, loss = local.train(client_num, copy.deepcopy(net_glob).to(args.device), neighbor_local)
                    local.weight = copy.deepcopy(w)

                elif args.method == "ours":
                    if epoch < args.warmup_epochs:
                        w, loss, expertise, inference = local.train_phase1(client_num,
                                                                           copy.deepcopy(net_glob).to(args.device))
                    else:
                        score_list = []
                        sim_list = []
                        inference_list[idx] = inference_list[idx].to(args.device)
                        for i in inference_list:
                            i = i.to(args.device)
                            sim_list.append(sim(i, inference_list[idx]))
                        for index, (e, s) in enumerate(zip(expertise_list, sim_list)):
                            if index != idx:
                                score = args.w_alpha * e + (1 - args.w_alpha) * s
                                score_list.append([score, index])
                        score_list.sort(key=lambda x: x[0], reverse=True)
                        neighbor1_score, neighbor2_score = score_list[0][0], score_list[1][0]
                        neighbor1_idx, neighbor2_idx = score_list[0][1], score_list[1][1]

                        w, loss, expertise, inference = local.train_phase2(
                            client_num,
                            copy.deepcopy(net_glob).to(args.device),
                            copy.deepcopy(local_update_objects[neighbor1_idx].net1).to(args.device),
                            copy.deepcopy(local_update_objects[neighbor2_idx].net1).to(args.device),
                            neighbor1_score,
                            neighbor2_score,
                        )
                    expertise_list[idx] = expertise
                    inference_list[idx] = inference

                else:
                    w, loss = local.train(client_num, copy.deepcopy(net_glob).to(args.device))

                # local_weights.update(idx, w, len(dict_users[idx]))
                local_weights.update(idx, w)
                local_losses.append(copy.deepcopy(loss))

        if epoch in args.loss_dist_epoch2:
            for client_num in range(args.num_users):
                local_update_objects[client_num].get_loss_dist(client_num=client_num, client=True, all_client=True)

        w_glob = local_weights.average()  # update global weights
        net_glob.load_state_dict(w_glob, strict=False)  # copy weight to net_glob

        local_weights.init()

        #         # Log - Train Accuracy
        #         if epoch == args.warmup_epochs and args.method == "global_GMM_base":
        #             result_dir = './save/{}/'.format(args.save_dir)
        #             local_epoch_backup = args.local_ep
        #             glob_w_backup = copy.deepcopy(net_glob.state_dict())

        #             with open(result_dir + "train_acc.csv", 'w', newline='') as train_acc_f:
        #                 train_acc_wr = csv.writer(train_acc_f)
        #                 train_acc_wr.writerow(["user_id", "train_acc", "noise_rate", "noise_type"])

        #                 for i in range(args.num_users):
        #                     local = local_update_objects[i]
        #                     w, loss = local.train(client_num, copy.deepcopy(net_glob).to(args.device))
        #                     net_glob.load_state_dict(w)

        #                     data_lst = []
        #                     for d_i in dict_users[i]:
        #                         data_lst.append([dataset_train[d_i][0], dataset_train[d_i][1]])

        #                     temp_data_loader = DataLoader(data_lst, batch_size=1, shuffle=False)
        #                     train_acc = test_img(net_glob, temp_data_loader, args)[0]
        #                     noise_type, noise_rate = user_noise_type_rates[i]
        #                     train_acc_wr.writerow([i, train_acc, noise_rate, noise_type])

        #             net_glob.load_state_dict(glob_w_backup)

        if args.method == 'RFL':
            sim = torch.nn.CosineSimilarity(dim=1)
            tmp = 0
            w_sum = 0
            for i in f_locals:
                sim_weight = sim(f_G, i).reshape(args.num_classes, 1)
                w_sum += sim_weight
                tmp += sim_weight * i
            for i in range(len(w_sum)):
                if w_sum[i] == 0:
                    print('check')
                    w_sum[i] = 1
            f_G = torch.div(tmp, w_sum)

        train_acc, train_loss = test_img(net_glob, log_train_data_loader, args)
        test_acc, test_loss = test_img(net_glob, log_test_data_loader, args)

        # for logging purposes
        results = dict(train_acc=train_acc, train_loss=train_loss,
                       test_acc=test_acc, test_loss=test_loss, )

        if args.send_2_models:
            w_glob2 = local_weights2.average()
            net_glob2.load_state_dict(w_glob2)
            local_weights2.init()
            # for logging purposes
            train_acc2, train_loss2 = test_img(net_glob2, log_train_data_loader, args)
            test_acc2, test_loss2 = test_img(net_glob2, log_test_data_loader, args)
            results2 = dict(train_acc2=train_acc2, train_loss2=train_loss2,
                            test_acc2=test_acc2, test_loss2=test_loss2, )
            results = {**results, **results2}

        print('Round {:3d}'.format(epoch))
        print(' - '.join([f'{k}: {v:.6f}' for k, v in results.items()]))

        logger.write(epoch=epoch + 1, **results)

        if epoch in args.loss_dist_epoch:
            if args.send_2_models:
                get_loss_dist(args, dataset_train, tmp_true_labels, net_glob, net_glob2)
            else:
                get_loss_dist(args, dataset_train, tmp_true_labels, net_glob)

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

    logger.close()
    noise_logger.close()
    print("time :", time.time() - start)
