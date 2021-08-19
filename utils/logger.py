import os
import csv
import numpy as np
import nsml
import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset


class Logger:
    def __init__(self, args, use_2_models):
        self.use_2_models = use_2_models

        # save results
        if args.save_dir is None:
            result_dir = './save/'
        else:
            result_dir = './save/{}/'.format(args.save_dir)

        result_f = 'nei[{}]_alpha[{}]_{}_{}_{}_{}_BS[{}]_LE[{}]_IID[{}]_NT[{}]_NGN[{}]_GNR[{}]_PT[{}]'.format(
            args.num_neighbors,
            args.w_alpha,
            args.dataset,
            args.method,
            args.model,
            args.epochs,
            args.local_bs,
            args.local_ep,
            args.iid,
            args.noise_type_lst,
            args.noise_group_num,
            args.group_noise_rate,
            args.partition,
        )

        if nsml.IS_ON_NSML:
            result_f = 'accuracy'
        else:
            result_f = 'nei[{}]_alpha[{}]_fedLNL_{}_{}_{}_{}_BS[{}]_LE[{}]_IID[{}]_NT[{}]_NGN[{}]_GNR[{}]_PT[{}]'.format(
                args.num_neighbors,
                args.w_alpha,
                args.dataset,
                args.method,
                args.model,
                args.epochs,
                args.local_bs,
                args.local_ep,
                args.iid,
                args.noise_type_lst,
                args.noise_group_num,
                args.group_noise_rate,
                args.partition,
            )

            if args.method in ['coteaching', 'coteaching+', 'finetune', 'lgfinetune', 'gfilter', 'gmix', 'lgteaching']:
                result_f += "_FR[{}]_FRS[{}]".format(args.forget_rate, args.forget_rate_schedule)

        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        self.f = open(result_dir + result_f + ".csv", 'w', newline='')
        self.wr = csv.writer(self.f)
        self.use_imagenetval = args.dataset == 'webvision'

        if self.use_2_models:
            columns = ['epoch', 'train_acc', 'train_loss', 'train_acc2', 'train_loss2',
                       'test_acc', 'test_loss', 'test_acc2', 'test_loss2']
        else:
            columns = ['epoch', 'train_acc', 'train_loss', 'test_acc', 'test_loss']

        if self.use_imagenetval:
            if self.use_2_models:
                columns.extend(['test_imagenetacc', 'test_imagenetloss', 'test_imagenetacc2', 'test_imagenetloss2'])
            else:
                columns.extend(['test_imagenetacc', 'test_imagenetloss'])

        self.wr.writerow(columns)

        # Option Save
        with open(result_dir + result_f + ".txt", 'w') as option_f:
            option_f.write(str(args))

    def write(self, epoch, train_acc, train_loss, test_acc, test_loss,
              train_acc2=None, train_loss2=None, test_acc2=None, test_loss2=None,
              test_imagenetacc=None, test_imagenetloss=None,
              test_imagenetacc2=None, test_imagenetloss2=None,
              ):

        if self.use_2_models:
            row = [epoch, train_acc, train_loss, train_acc2, train_loss2,
                   test_acc, test_loss, test_acc2, test_loss2]
        else:
            row = [epoch, train_acc, train_loss, test_acc, test_loss]

        if self.use_imagenetval:
            if self.use_2_models:
                row.extend([test_imagenetacc, test_imagenetloss, test_imagenetacc2, test_imagenetloss2])
            else:
                row.extend([test_imagenetacc, test_imagenetloss])

        self.wr.writerow(row)

    def close(self):
        self.f.close()


class NoiseLogger:
    def __init__(self, args, user_noisy_data, dict_users):
        # save results
        if args.save_dir is None:
            result_dir = './save/'
        else:
            result_dir = './save/{}/'.format(args.save_dir)

        if nsml.IS_ON_NSML:
            result_f = 'noise_label_accuracy'
        else:
            result_f = 'nei[{}]_alpha[{}]_fedLNL_noise_{}_{}_{}_{}_BS[{}]_LE[{}]_IID[{}]_NT[{}]_NGN[{}]_GNR[{}]_PT[{}]'.format(
                args.num_neighbors,
                args.w_alpha,
                args.dataset,
                args.method,
                args.model,
                args.epochs,
                args.local_bs,
                args.local_ep,
                args.iid,
                args.noise_type_lst,
                args.noise_group_num,
                args.group_noise_rate,
                args.partition,
            )

        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        self.f = open(result_dir + result_f + ".csv", 'w', newline='')
        self.wr = csv.writer(self.f)

        self.wr.writerow(['epoch', 'user_id', 'total_epochs', 'TP', 'FP', 'FN', 'TN', 'precision', 'recall'])

        self.user_total_data = dict_users
        self.user_noisy_data = user_noisy_data
        self.user_clean_data = {user: list(set(total_data) - set(user_noisy_data[user]))
                                for user, total_data in dict_users.items()}

        self.clean_ids = []

    def update(self, ids):
        self.clean_ids.extend(ids)

    def write(self, epoch, user_id, total_epochs):
        # precision = # of clean examples in selected set / # of total selected set
        # recall = # of clean examples in selected set / # of total clean in data

        clean_data = set(self.user_clean_data[user_id])
        noisy_data = set(self.user_noisy_data[user_id])

        if self.clean_ids:
            self.clean_ids = np.array(self.clean_ids)
            pred_clean_data = set(self.clean_ids)
        else:
            pred_clean_data = set()

        pred_noisy_data = set(self.user_total_data[user_id]) - set(self.clean_ids)

        true_positive = len(set(pred_clean_data) & set(clean_data))
        false_positive = len(set(pred_clean_data) & set(noisy_data))
        false_negative = len(set(pred_noisy_data) & set(clean_data))
        true_negative = len(set(pred_noisy_data) & set(noisy_data))

        precision = true_positive / max(true_positive + false_positive, 1)
        recall = true_positive / max(true_positive + false_negative, 1)

        self.wr.writerow(
            [epoch, user_id, total_epochs, true_positive, false_positive, false_negative, true_negative, precision,
             recall])

        self.clean_ids = []

    def close(self):
        self.f.close()


def get_loss_dist(args, dataset, tmp_true_labels, net1, net2=None, client_num=None, client=False):
    if args.save_dir2 is None:
        result_dir = './save/'
    else:
        result_dir = './save/{}/'.format(args.save_dir2)

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    if nsml.IS_ON_NSML:
        if client:
            result_f = 'client_loss_dist'
        else:
            result_f = 'loss_dist'
    else:
        if client:
            result_f = 'client_loss_dist_model[{}]_method[{}]_noise{}_NR{}_IID[{}]'.format(
                args.model,
                args.method,
                args.noise_type_lst,
                args.group_noise_rate,
                args.iid)
        else:
            result_f = 'loss_dist_model[{}]_method[{}]_noise{}_NR{}_IID[{}]'.format(
                args.model,
                args.method,
                args.noise_type_lst,
                args.group_noise_rate,
                args.iid,
            )

    f = open(result_dir + result_f + ".csv", 'a', newline='')
    wr = csv.writer(f)

    if args.g_epoch == args.loss_dist_epoch[0]:
        if client:
            if client_num == 0:
                wr.writerow(['epoch', 'client_num', 'data_idx', 'is_noise', 'loss'])
            else:
                pass
        else:
            wr.writerow(['epoch', 'data_idx', 'is_noise', 'loss'])

    net1.eval()
    if args.send_2_models:
        net2.eval()

    ce = nn.CrossEntropyLoss(reduce=False)
    with torch.no_grad():
        for batch_idx, batch in enumerate(DataLoader(dataset, batch_size=1, shuffle=False)):
            if client:
                images, labels, _, real_idx = batch
            else:
                images, labels = batch

            images, labels = images.to(args.device), labels.to(args.device)

            if args.send_2_models:
                logits1 = net1(images)
                logits2 = net2(images)
                loss1 = ce(logits1, labels)
                loss2 = ce(logits2, labels)
                loss = (loss1 + loss2) / 2
            else:
                if args.feature_return:
                    logits, feature = net1(images)
                else:
                    logits = net1(images)
                loss = ce(logits, labels)

            if client:
                if tmp_true_labels[real_idx] != labels:
                    is_noise = 1
                else:
                    is_noise = 0
                wr.writerow([args.g_epoch, client_num, real_idx.item(), is_noise, loss.item()])
            else:
                if tmp_true_labels[batch_idx] != labels:
                    is_noise = 1
                else:
                    is_noise = 0
                wr.writerow([args.g_epoch, batch_idx, is_noise, loss.item()])
    f.close()
