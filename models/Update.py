#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import copy
from sklearn.mixture import GaussianMixture
from .correctors import SelfieCorrector, JointOptimCorrector
import os
import csv
from .Nets import get_model
import nsml
#from utils.logger import get_loss_dist


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs, idx_return=False, real_idx_return=False):
        self.dataset = dataset
        self.idxs = list(idxs)
        self.idx_return = idx_return
        self.real_idx_return = real_idx_return

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        item = int(item)
        image, label = self.dataset[self.idxs[item]]

        if self.idx_return:
            return image, label, item
        elif self.real_idx_return:
            return image, label, item, self.idxs[item]
        else:
            return image, label


class DatasetSplitRFL(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        item = int(item)
        image, label = self.dataset[self.idxs[item]]

        return image, label, self.idxs[item]


class DatasetSplitHS(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        item = int(item)
        img, target, soft_label, prediction, index = self.dataset[self.idxs[item]]

        return img, target, soft_label, prediction, index


class PairProbDataset(Dataset):
    def __init__(self, dataset, idxs, prob, idx_return=False):
        self.dataset = dataset
        self.idxs = list(idxs)
        self.idx_return = idx_return
        self.prob = prob

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        item = int(item)
        image1, label = self.dataset[self.idxs[item]]
        image2, label = self.dataset[self.idxs[item]]
        prob = self.prob[self.idxs[item]]

        if self.idx_return:
            return image1, image2, label, prob, item
        else:
            return image1, image2, label, prob


class PairDataset(Dataset):
    def __init__(self, dataset, idxs, idx_return=False, label_return=False):
        self.dataset = dataset
        self.idxs = list(idxs)
        self.idx_return = idx_return
        self.label_return = label_return

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        item = int(item)
        image1, label = self.dataset[self.idxs[item]]
        image2, label = self.dataset[self.idxs[item]]
        sample = (image1, image2,)

        if self.label_return:
            sample += (label,)

        if self.idx_return:
            sample += (item,)

        return sample


def mixup(inputs, targets, alpha=1.0):
    l = np.random.beta(alpha, alpha)
    l = max(l, 1 - l)

    idx = torch.randperm(inputs.size(0))

    input_a, input_b = inputs, inputs[idx]
    target_a, target_b = targets, targets[idx]

    mixed_input = l * input_a + (1 - l) * input_b
    mixed_target = l * target_a + (1 - l) * target_b

    return mixed_input, mixed_target


def linear_rampup(current, warm_up, lambda_u, rampup_length=16):
    current = np.clip((current - warm_up) / rampup_length, 0.0, 1.0)
    return lambda_u * float(current)


class SemiLoss:
    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, lambda_u, epoch, warm_up):
        probs_u = torch.softmax(outputs_u, dim=1)

        # labeled data loss
        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        # unlabeled data loss
        Lu = torch.mean((probs_u - targets_u) ** 2)

        lamb = linear_rampup(epoch, warm_up, lambda_u)

        return Lx + lamb * Lu


def get_local_update_objects(args, dataset_train, dict_users=None, noise_rates=None, net_glob=None, net_local_lst=None,
                             noise_logger=None, tmp_true_labels=None):
    local_update_objects = []
    for idx, noise_rate in zip(range(args.num_users), noise_rates):
        local_update_args = dict(
            args=args,
            user_idx=idx,
            dataset=dataset_train,
            idxs=dict_users[idx],
            noise_logger=noise_logger,
            tmp_true_labels=tmp_true_labels
        )

        if args.method == 'default':
            local_update_object = BaseLocalUpdate(**local_update_args)

        elif args.method == 'babu':
            local_update_object = BaseLocalUpdate(is_babu=True, **local_update_args)

        elif args.method == 'selfie':
            local_update_object = LocalUpdateSELFIE(noise_rate=noise_rate, **local_update_args)

        elif args.method == 'jointoptim':
            local_update_object = LocalUpdateJointOptim(**local_update_args)

        elif args.method in ['coteaching', 'coteaching+']:
            local_update_object = LocalUpdateCoteaching(is_coteaching_plus=bool(args.method == 'coteaching+'),
                                                        **local_update_args)
        elif args.method == 'dividemix':
            local_update_object = LocalUpdateDivideMix(**local_update_args)

        elif args.method == 'gfilter':
            local_update_object = LocalUpdateGFilter(**local_update_args)

        elif args.method == 'gmix':
            local_update_object = LocalUpdateGMix(**local_update_args)

        elif args.method == 'lgfinetune':
            local_update_object = LocalUpdateLGFineTuning(l_net=net_glob, **local_update_args)

        elif args.method == 'finetune':
            local_update_object = LocalUpdateFinetuning(**local_update_args)

        elif args.method == 'history':
            local_update_object = LocalUpdateHS(**local_update_args)

        elif args.method == 'lgteaching':
            local_update_object = LocalUpdateLGteaching(l_net=net_glob, **local_update_args)

        elif args.method == 'fedprox':
            local_update_object = LocalUpdateFedProx(**local_update_args)

        elif args.method == 'lgcorrection':
            local_update_object = LocalUpdateLGCorrection(noise_rate=noise_rate, l_net=net_local_lst[idx],
                                                          **local_update_args)

        elif args.method == 'RFL':
            local_update_object = LocalUpdateRFL(**local_update_args)

        local_update_objects.append(local_update_object)

    return local_update_objects


class BaseLocalUpdate:
    def __init__(
            self,
            args,
            user_idx=None,
            dataset=None,
            idxs=None,
            idx_return=False,
            real_idx_return=False,
            is_babu=False,
            noise_logger=None,
            tmp_true_labels=None
    ):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        
        self.dataset = dataset
        self.idxs = idxs
        self.user_idx = user_idx
        self.noise_logger = noise_logger

        self.idx_return = idx_return
        self.real_idx_return = real_idx_return

        self.ldr_train = DataLoader(
            DatasetSplit(dataset, idxs, idx_return=idx_return, real_idx_return=real_idx_return),
            batch_size=self.args.local_bs,
            shuffle=True,
        )
        self.total_epochs = 0
        self.epoch = 0
        self.batch_idx = 0
        self.is_babu = is_babu

        self.tmp_true_labels = tmp_true_labels
        
        self.net1 = get_model(self.args)
        self.net2 = get_model(self.args)
        self.net1 = self.net1.to(self.args.device)
        self.net2 = self.net2.to(self.args.device)
        
    def update_label_accuracy(self):
        self.noise_logger.write(
            epoch=self.args.g_epoch,
            user_id=self.user_idx,
            total_epochs=self.total_epochs,
        )

    def train(self, client_num, net, net2=None):
        if net2 is None:
            return self.train_single_model(client_num, net)
        else:
            return self.train_multiple_models(client_num, net, net2)

        
    def get_loss_dist(self, client_num=None, client=True, all_client=False):
        dataset = DatasetSplit(self.dataset, self.idxs, real_idx_return=True)
   
        if self.args.save_dir2 is None:
            result_dir = './save/'
        else:
            result_dir = './save/{}/'.format(self.args.save_dir2)

        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        if nsml.IS_ON_NSML:
            if client:
                result_f = 'client_loss_dist'
            else:
                result_f = 'loss_dist'
        else:
            if client:
                if all_client:
                    result_f = 'all_client_loss_dist_model[{}]_method[{}]_noise{}_NR{}_IID[{}]'.format(
                        self.args.model,
                        self.args.method,
                        self.args.noise_type_lst,
                        self.args.group_noise_rate,
                        self.args.iid)
                else:
                    result_f = 'client_loss_dist_model[{}]_method[{}]_noise{}_NR{}_IID[{}]'.format(
                        self.args.model,
                        self.args.method,
                        self.args.noise_type_lst,
                        self.args.group_noise_rate,
                        self.args.iid)
            else:
                result_f = 'loss_dist_model[{}]_method[{}]_noise{}_NR{}_IID[{}]'.format(
                    self.args.model,
                    self.args.method,
                    self.args.noise_type_lst,
                    self.args.group_noise_rate,
                    self.args.iid,
                )

        f = open(result_dir + result_f + ".csv", 'a', newline='')
        wr = csv.writer(f)
        
        if all_client:
            if self.args.g_epoch == self.args.loss_dist_epoch2[0]:
                if client_num == 0:
                    wr.writerow(['epoch', 'client_num', 'data_idx', 'is_noise', 'loss'])
                else:
                    pass
        else:
            if self.args.g_epoch == self.args.loss_dist_epoch[0]:
                if client:
                    if client_num == 0:
                        wr.writerow(['epoch', 'client_num', 'data_idx', 'is_noise', 'loss'])
                    else:
                        pass
                else:
                    wr.writerow(['epoch', 'data_idx', 'is_noise', 'loss'])

        self.net1.eval()
        if self.args.send_2_models:
            self.net2.eval()

        ce = nn.CrossEntropyLoss(reduce=False)
        with torch.no_grad():
            for batch_idx, batch in enumerate(DataLoader(dataset, batch_size=1, shuffle=False)):
                if client:
                    images, labels, _, real_idx = batch
                else:
                    images, labels = batch

                images, labels = images.to(self.args.device), labels.to(self.args.device)

                if self.args.send_2_models:
                    logits1 = self.net1(images)
                    logits2 = self.net2(images)
                    loss1 = ce(logits1, labels)
                    loss2 = ce(logits2, labels)
                    loss = (loss1 + loss2) / 2
                else:
                    if self.args.feature_return:
                        logits, feature = self.net1(images)
                    else:
                        logits = self.net1(images)
                    loss = ce(logits, labels)

                if client:
                    if self.tmp_true_labels[real_idx] != labels:
                        is_noise = 1
                    else:
                        is_noise = 0
                    wr.writerow([self.args.g_epoch, client_num, real_idx.item(), is_noise, loss.item()])
                else:
                    if self.tmp_true_labels[batch_idx] != labels:
                        is_noise = 1
                    else:
                        is_noise = 0
                    wr.writerow([self.args.g_epoch, batch_idx, is_noise, loss.item()])
        f.close()


    def train_single_model(self, client_num, net):
        net.train()

        if self.is_babu:
            body_params = [p for name, p in net.named_parameters() if 'linear' not in name]
            head_params = [p for name, p in net.named_parameters() if 'linear' in name]

            optimizer = torch.optim.SGD([{'params': body_params, 'lr': self.args.lr, 'momentum': self.args.momentum},
                                         {'params': head_params, 'lr': 0.0}])
        else:
            optimizer = torch.optim.SGD(
                net.parameters(),
                lr=self.args.lr,
                momentum=self.args.momentum,
                weight_decay=self.args.weight_decay,
            )

        epoch_loss = []
        for epoch in range(self.args.local_ep):
            self.epoch = epoch
            batch_loss = []
            for batch_idx, batch in enumerate(self.ldr_train):
                self.batch_idx = batch_idx
                net.zero_grad()

                loss = self.forward_pass(batch, net)
                loss.backward()
                optimizer.step()

                if self.args.verbose and batch_idx % 10 == 0:
                    print(f"Update Epoch: {epoch} [{batch_idx}/{len(self.ldr_train)}"
                          f"({100. * batch_idx / len(self.ldr_train):.0f}%)]\tLoss: {loss.item():.6f}")

                batch_loss.append(loss.item())
                self.on_batch_end()

            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            self.total_epochs += 1
            self.on_epoch_end()
        
        self.net1.load_state_dict(net.state_dict())    
        
        if self.args.g_epoch in self.args.loss_dist_epoch:
            self.get_loss_dist(client_num=client_num, client=True)

        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def train_multiple_models(self, client_num, net1, net2):
        net1.train()
        net2.train()

        optimizer_args = dict(
            lr=self.args.lr,
            momentum=self.args.momentum,
            weight_decay=self.args.weight_decay,
        )
        optimizer1 = torch.optim.SGD(net1.parameters(), **optimizer_args)
        optimizer2 = torch.optim.SGD(net2.parameters(), **optimizer_args)

        epoch_loss1 = []
        epoch_loss2 = []
        for epoch in range(self.args.local_ep):
            self.epoch = epoch
            batch_loss1 = []
            batch_loss2 = []
            for batch_idx, batch in enumerate(self.ldr_train):
                self.batch_idx = batch_idx
                net1.zero_grad()
                net2.zero_grad()

                loss1, loss2 = self.forward_pass(batch, net1, net2)
                loss1.backward()
                loss2.backward()
                optimizer1.step()
                optimizer2.step()

                if self.args.verbose and batch_idx % 10 == 0:
                    print(f"Update Epoch: {epoch} [{batch_idx}/{len(self.ldr_train)}"
                          f"({100. * batch_idx / len(self.ldr_train):.0f}%)]\tLoss: {loss1.item():.6f}"
                          f"\tLoss: {loss2.item():.6f}")

                batch_loss1.append(loss1.item())
                batch_loss2.append(loss2.item())
                self.on_batch_end()

            epoch_loss1.append(sum(batch_loss1) / len(batch_loss1))
            epoch_loss2.append(sum(batch_loss2) / len(batch_loss2))
            self.total_epochs += 1
            self.on_epoch_end()
        
        
        self.net1.load_state_dict(net1.state_dict())   
        self.net2.load_state_dict(net2.state_dict())   
        
        if self.args.g_epoch in self.args.loss_dist_epoch:
            self.get_loss_dist(client_num=client_num, client=True)

        return net1.state_dict(), sum(epoch_loss1) / len(epoch_loss1), \
               net2.state_dict(), sum(epoch_loss2) / len(epoch_loss2)

    def forward_pass(self, batch, net, net2=None):
        if self.idx_return:
            images, labels, _ = batch

        elif self.real_idx_return:
            images, labels, _, ids = batch
            self.noise_logger.update(ids)

        else:
            images, labels = batch

        images = images.to(self.args.device)
        labels = labels.to(self.args.device)

        log_probs = net(images)
        loss = self.loss_func(log_probs, labels)
        if net2 is None:
            return loss

        # 2 models
        log_probs2 = net2(images)
        loss2 = self.loss_func(log_probs2, labels)
        return loss, loss2

    def on_batch_end(self):
        pass

    def on_epoch_end(self):
        self.update_label_accuracy()


class LocalUpdateRFL(BaseLocalUpdate):
    def __init__(self, args, dataset=None, user_idx=None, idxs=None, noise_logger=None, tmp_true_labels=None):
        super().__init__(
            args=args,
            dataset=dataset,
            user_idx=user_idx,
            idxs=idxs,
            noise_logger=noise_logger
        )
        self.pseudo_labels = torch.zeros(len(self.dataset), dtype=torch.long, device=self.args.device)
        self.sim = torch.nn.CosineSimilarity(dim=1)
        self.loss_func = torch.nn.CrossEntropyLoss(reduce=False)
        self.ldr_train = DataLoader(
            DatasetSplitRFL(dataset, idxs),
            batch_size=self.args.local_bs,
            shuffle=True,
        )
        self.ldr_train_tmp = DataLoader(DatasetSplitRFL(dataset, idxs), batch_size=1, shuffle=True)
        self.tmp_true_labels = tmp_true_labels

    def RFLloss(self, logit, labels, feature, f_k, mask, small_loss_idxs, lambda_cen, lambda_e, new_labels):
        mse = torch.nn.MSELoss(reduce=False)
        ce = torch.nn.CrossEntropyLoss()
        sm = torch.nn.Softmax()
        lsm = torch.nn.LogSoftmax()

        L_c = ce(logit[small_loss_idxs], new_labels)
        L_cen = torch.sum(
            mask[small_loss_idxs] * torch.sum(mse(feature[small_loss_idxs], f_k[labels[small_loss_idxs]]), 1))
        L_e = -torch.mean(torch.sum(sm(logit[small_loss_idxs]) * lsm(logit[small_loss_idxs]), dim=1))

        if self.args.g_epoch < 100:
            lambda_cen = 0.01 * (self.args.g_epoch + 1)

        # return L_c + lambda_e * L_e
        return L_c + lambda_cen * L_cen + lambda_e * L_e

    def get_small_loss_samples(self, y_pred, y_true, forget_rate):
        loss = self.loss_func(y_pred, y_true)
        ind_sorted = np.argsort(loss.data.cpu()).cuda()
        loss_sorted = loss[ind_sorted]

        remember_rate = 1 - forget_rate
        num_remember = int(remember_rate * len(loss_sorted))

        ind_update = ind_sorted[:num_remember]

        return ind_update

    def train(self, net, f_G, client_num):
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum,
                                    weight_decay=self.args.weight_decay)
        epoch_loss = []

        net.eval()
        f_k = torch.zeros(self.args.num_classes, self.args.feature_dim, device=self.args.device)
        n_labels = torch.zeros(self.args.num_classes, 1, device=self.args.device)

        # obtain global-guided pseudo labels y_hat by y_hat_k = C_G(F_G(x_k))
        # initialization of global centroids
        # obtain naive average feature
        with torch.no_grad():
            for batch_idx, (images, labels, idxs) in enumerate(self.ldr_train_tmp):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                logit, feature = net(images)
                self.pseudo_labels[idxs] = torch.argmax(logit)
                if self.args.g_epoch == 0:
                    f_k[labels] += feature
                    n_labels[labels] += 1

        if self.args.g_epoch == 0:
            for i in range(len(n_labels)):
                if n_labels[i] == 0:
                    n_labels[i] = 1
            f_k = torch.div(f_k, n_labels)
        else:
            f_k = f_G

        net.train()
        for iter in range(self.args.local_ep):
            batch_loss = []
            correct_num = 0
            total = 0
            for batch_idx, batch in enumerate(self.ldr_train):
                net.zero_grad()
                images, labels, idx = batch
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                logit, feature = net(images)
                feature = feature.detach()
                f_k = f_k.to(self.args.device)

                # batch 안에서의 index 순서
                small_loss_idxs = self.get_small_loss_samples(logit, labels, self.args.forget_rate)

                y_k_tilde = torch.zeros(self.args.local_bs, device=self.args.device)
                mask = torch.zeros(self.args.local_bs, device=self.args.device)
                for i in small_loss_idxs:
                    y_k_tilde[i] = torch.argmax(self.sim(f_k, torch.reshape(feature[i], (1, self.args.feature_dim))))
                    if y_k_tilde[i] == labels[i]:
                        mask[i] = 1

                if client_num == 0 and iter == 0:
                    if batch_idx == 0:
                        print('============================================================================')
                        print('------------------------------------mask-----------------------------------\n',
                              mask[small_loss_idxs])
                        print('---------------------------------y_k_tilde---------------------------------\n',
                              y_k_tilde[small_loss_idxs])
                        print('----------------------------------labels-----------------------------------\n',
                              labels[small_loss_idxs])
                        print('-------------------------------pseudo_labels-------------------------------\n',
                              self.pseudo_labels[idx[small_loss_idxs]])
                        print('--------------------------------true_labels--------------------------------\n',
                              self.tmp_true_labels[idx[small_loss_idxs]])

                    total += len(small_loss_idxs)
                    for i in idx[small_loss_idxs]:
                        if self.pseudo_labels[i] == self.tmp_true_labels[i]:
                            correct_num += 1

                            # When to use pseudo-labels
                if self.args.g_epoch < self.args.T_pl:
                    for i in small_loss_idxs:
                        self.pseudo_labels[idx[i]] = labels[i]

                # For loss calculating
                new_labels = mask[small_loss_idxs] * labels[small_loss_idxs] + (1 - mask[small_loss_idxs]) * \
                             self.pseudo_labels[idx[small_loss_idxs]]
                new_labels = new_labels.type(torch.LongTensor).to(self.args.device)

                if client_num == 0 and iter == 0 and batch_idx == 0:
                    print('--------------------------------new_labels---------------------------------\n', new_labels)
                    print('============================================================================')

                loss = self.RFLloss(logit, labels, feature, f_k, mask, small_loss_idxs, self.args.lambda_cen,
                                    self.args.lambda_e, new_labels)

                # weight update by minimizing loss: L_total = L_c + lambda_cen * L_cen + lambda_e * L_e
                loss.backward()
                optimizer.step()

                # obtain loss based average features f_k,j_hat from small loss dataset
                f_kj_hat = torch.zeros(self.args.num_classes, self.args.feature_dim, device=self.args.device)
                n = torch.zeros(self.args.num_classes, 1, device=self.args.device)
                for i in small_loss_idxs:
                    f_kj_hat[labels[i]] += feature[i]
                    n[labels[i]] += 1
                for i in range(len(n)):
                    if n[i] == 0:
                        n[i] = 1
                f_kj_hat = torch.div(f_kj_hat, n)

                # update local centroid f_k
                one = torch.ones(self.args.num_classes, 1, device=self.args.device)
                f_k = (one - self.sim(f_k, f_kj_hat).reshape(self.args.num_classes, 1) ** 2) * f_k + (
                            self.sim(f_k, f_kj_hat).reshape(self.args.num_classes, 1) ** 2) * f_kj_hat

                if self.args.verbose and batch_idx % 10 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(images), len(self.ldr_train.dataset),
                              100. * batch_idx / len(self.ldr_train), loss.item()))
                batch_loss.append(loss.item())

            if client_num == 0 and iter == 0:
                print('<<<<<<<<<<pseudo labeing accuracy>>>>>>>>>>: {}'.format(100 * correct_num / total))

            epoch_loss.append(sum(batch_loss) / len(batch_loss))
        
        
        self.net1.load_state_dict(net.state_dict())   
        
        if self.args.g_epoch == 100:
            self.get_loss_dist(client_num=client_num, client=True)

        return net.state_dict(), sum(epoch_loss) / len(epoch_loss), f_k


class LocalUpdateFedProx(BaseLocalUpdate):
    def __init__(self, args, user_idx=None, dataset=None, idxs=None, noise_logger=None):
        super().__init__(
            args=args,
            user_idx=user_idx,
            dataset=dataset,
            idxs=idxs,
            noise_logger=noise_logger,
        )
        self.glob_net = None
        self.fed_prox_mu = args.init_fed_prox_mu

    def forward_pass(self, batch, net, net2=None):
        if self.epoch == 0 and self.batch_idx == 0:
            self.glob_net = copy.deepcopy(net)

            if self.args.g_epoch == int(self.args.epochs * 0.7):
                self.fed_prox_mu *= 10

        images, labels = batch
        images = images.to(self.args.device)
        labels = labels.to(self.args.device)

        log_probs = net(images)
        loss = self.loss_func(log_probs, labels)

        # for fedprox
        fed_prox_reg = 0.0
        for l_param, g_param in zip(net.parameters(), self.glob_net.parameters()):
            fed_prox_reg += (self.fed_prox_mu / 2 * torch.norm((l_param - g_param)) ** 2)
        loss += fed_prox_reg

        return loss


class LocalUpdateSELFIE(BaseLocalUpdate):
    def __init__(self, args, user_idx=None, dataset=None, idxs=None, noise_rate=0, noise_logger=None,
                 tmp_true_labels=None):
        super().__init__(
            args=args,
            user_idx=user_idx,
            dataset=dataset,
            idxs=idxs,
            real_idx_return=True,
            noise_logger=noise_logger,
            tmp_true_labels=tmp_true_labels
        )

        self.loss_func = nn.CrossEntropyLoss(reduction='none')
        self.total_epochs = 0
        self.warmup = args.warmup_epochs
        self.corrector = SelfieCorrector(
            queue_size=args.queue_size,
            uncertainty_threshold=args.uncertainty_threshold,
            noise_rate=noise_rate,
            num_classes=args.num_classes,
        )

    def forward_pass(self, batch, net, net2=None):
        images, labels, _, ids = batch
        images = images.to(self.args.device)
        labels = labels.to(self.args.device)
        ids = ids.numpy()

        log_probs = net(images)
        loss_array = self.loss_func(log_probs, labels)

        # update prediction history
        self.corrector.update_prediction_history(
            ids=ids,
            outputs=log_probs.cpu().detach().numpy(),
        )

        if self.args.g_epoch >= self.args.warmup_epochs:
            # correct labels, remove noisy data
            images, labels, ids = self.corrector.patch_clean_with_corrected_sample_batch(
                ids=ids,
                X=images,
                y=labels,
                loss_array=loss_array.cpu().detach().numpy(),
            )
            images = images.to(self.args.device)
            labels = labels.to(self.args.device)
            log_probs = net(images)
            loss_array = self.loss_func(log_probs, labels)

        self.noise_logger.update(ids)
        loss = loss_array.mean()
        return loss


class LocalUpdateJointOptim(BaseLocalUpdate):
    def __init__(self, args, user_idx=None, dataset=None, idxs=None, noise_logger=None, tmp_true_labels=None):
        super().__init__(
            args=args,
            user_idx=user_idx,
            dataset=dataset,
            idxs=idxs,
            real_idx_return=True,
            noise_logger=noise_logger,
            tmp_true_labels=tmp_true_labels
        )
        self.corrector = JointOptimCorrector(
            queue_size=args.queue_size,
            num_classes=args.num_classes,
            data_size=len(idxs),
        )

    def forward_pass(self, batch, net, net2=None):
        images, labels, _, ids = batch
        ids = ids.numpy()

        hard_labels, soft_labels = self.corrector.get_labels(ids, labels)
        if self.args.labeling == 'soft':
            labels = soft_labels.to(self.args.device)
        else:
            labels = hard_labels.to(self.args.device)
        images = images.to(self.args.device)

        logits = net(images)
        probs = F.softmax(logits, dim=1)

        loss = self.joint_optim_loss(logits, probs, labels)
        self.corrector.update_probability_history(ids, probs.cpu().detach())
        self.noise_logger.update(ids)

        return loss

    def on_epoch_end(self):
        if self.args.g_epoch >= self.args.warmup_epochs:
            self.corrector.update_labels()

    def joint_optim_loss(self, logits, probs, soft_targets, is_cross_entropy=False):
        if is_cross_entropy:
            loss = -torch.mean(torch.sum(F.log_softmax(logits, dim=1) * soft_targets, dim=1))

        else:
            # We introduce a prior probability distribution p,
            # which is a distribution of classes among all training data.
            p = torch.ones(self.args.num_classes, device=self.args.device) / self.args.num_classes

            avg_probs = torch.mean(probs, dim=0)

            L_c = -torch.mean(torch.sum(F.log_softmax(logits, dim=1) * soft_targets, dim=1))
            L_p = -torch.sum(torch.log(avg_probs) * p)
            L_e = -torch.mean(torch.sum(F.log_softmax(logits, dim=1) * probs, dim=1))

            loss = L_c + self.args.alpha * L_p + self.args.beta * L_e

        return loss


class LocalUpdateGFilter(BaseLocalUpdate):
    def __init__(self, args, user_idx=None, dataset=None, idxs=None, noise_logger=None):
        super().__init__(
            args=args,
            user_idx=user_idx,
            dataset=dataset,
            idxs=idxs,
            real_idx_return=True,
            noise_logger=noise_logger,
        )
        self.loss_func = nn.CrossEntropyLoss(reduce=False)

    def forward_pass(self, batch, net, net2=None):
        images, labels, _, ids = batch
        images = images.to(self.args.device)
        labels = labels.to(self.args.device)

        log_probs = net(images)
        loss, indices = self.filter_loss(log_probs, labels, self.args.forget_rate)
        self.noise_logger.update(ids[indices])
        return loss

    def filter_loss(self, y_pred, y_true, forget_rate):
        loss = self.loss_func(y_pred, y_true)
        ind_sorted = np.argsort(loss.data.cpu()).cuda()
        loss_sorted = loss[ind_sorted]

        remember_rate = 1 - forget_rate
        num_remember = int(remember_rate * len(loss_sorted))

        ind_update = ind_sorted[:num_remember]
        loss_update = self.loss_func(y_pred[ind_update], y_true[ind_update])

        return torch.sum(loss_update) / num_remember, ind_update


class LocalUpdateCoteaching(BaseLocalUpdate):
    def __init__(self, args, user_idx=None, dataset=None, idxs=None, noise_logger=None, is_coteaching_plus=False,
                 tmp_true_labels=False):
        super().__init__(
            args=args,
            user_idx=user_idx,
            dataset=dataset,
            idxs=idxs,
            real_idx_return=True,
            noise_logger=noise_logger,
            tmp_true_labels=tmp_true_labels
        )
        self.loss_func = nn.CrossEntropyLoss(reduce=False)
        self.is_coteaching_plus = is_coteaching_plus

        self.init_epoch = 10  # only used for coteaching+

    def forward_pass(self, batch, net, net2=None):
        images, labels, indices, ids = batch

        images = images.to(self.args.device)
        labels = labels.to(self.args.device)
        log_probs1 = net(images)
        log_probs2 = net2(images)

        loss_args = dict(
            y_pred1=log_probs1,
            y_pred2=log_probs2,
            y_true=labels,
            forget_rate=self.args.forget_rate,
        )

        if self.is_coteaching_plus and self.epoch >= self.init_epoch:
            loss1, loss2, indices = self.loss_coteaching_plus(
                indices=indices, step=self.epoch * self.batch_idx, **loss_args)
        else:
            loss1, loss2, indices = self.loss_coteaching(**loss_args)

        self.noise_logger.update(ids[indices])
        return loss1, loss2

    def loss_coteaching(self, y_pred1, y_pred2, y_true, forget_rate):
        loss_1 = self.loss_func(y_pred1, y_true)
        ind_1_sorted = torch.argsort(loss_1)

        loss_2 = self.loss_func(y_pred2, y_true)
        ind_2_sorted = torch.argsort(loss_2)

        remember_rate = 1 - forget_rate
        num_remember = int(remember_rate * len(ind_1_sorted))

        ind_1_update = ind_1_sorted[:num_remember]
        ind_2_update = ind_2_sorted[:num_remember]
        # exchange
        loss_1_update = self.loss_func(y_pred1[ind_2_update], y_true[ind_2_update])
        loss_2_update = self.loss_func(y_pred2[ind_1_update], y_true[ind_1_update])

        ind_1_update = list(ind_1_update.cpu().detach().numpy())

        return torch.sum(loss_1_update) / num_remember, torch.sum(loss_2_update) / num_remember, ind_1_update

    def loss_coteaching_plus(self, y_pred1, y_pred2, y_true, forget_rate, indices, step):
        outputs = F.softmax(y_pred1, dim=1)
        outputs2 = F.softmax(y_pred2, dim=1)

        _, pred1 = torch.max(y_pred1.data, 1)
        _, pred2 = torch.max(y_pred2.data, 1)

        pred1, pred2 = pred1.cpu().numpy(), pred2.cpu().numpy()

        logical_disagree_id = np.zeros(y_true.size(), dtype=bool)
        disagree_id = []
        for idx, p1 in enumerate(pred1):
            if p1 != pred2[idx]:
                disagree_id.append(idx)
                logical_disagree_id[idx] = True

        temp_disagree = indices * logical_disagree_id.astype(np.int64)
        ind_disagree = np.asarray([i for i in temp_disagree if i != 0]).transpose()
        try:
            assert ind_disagree.shape[0] == len(disagree_id)
        except:
            disagree_id = disagree_id[:ind_disagree.shape[0]]

        if len(disagree_id) > 0:
            update_labels = y_true[disagree_id]
            update_outputs = outputs[disagree_id]
            update_outputs2 = outputs2[disagree_id]
            loss_1, loss_2, indices = self.loss_coteaching(update_outputs, update_outputs2, update_labels, forget_rate)
        else:
            update_step = np.logical_or(logical_disagree_id, step < 5000).astype(np.float32)
            update_step = Variable(torch.from_numpy(update_step)).cuda()

            cross_entropy_1 = F.cross_entropy(outputs, y_true)
            cross_entropy_2 = F.cross_entropy(outputs2, y_true)

            loss_1 = torch.sum(update_step * cross_entropy_1) / y_true.size()[0]
            loss_2 = torch.sum(update_step * cross_entropy_2) / y_true.size()[0]
            indices = range(y_true.size()[0])
        return loss_1, loss_2, indices


class LocalUpdateLGteaching(LocalUpdateCoteaching):
    def __init__(self, args, user_idx=None, dataset=None, idxs=None, noise_logger=None, l_net=None):
        super().__init__(
            args=args,
            user_idx=user_idx,
            dataset=dataset,
            idxs=idxs,
            noise_logger=noise_logger,
        )
        self.l_net = copy.deepcopy(l_net)

    def train(self, net, net2=None):
        w_g, loss1, w_l, loss2 = self.train_multiple_models(net, self.l_net)
        return w_g, loss1


class LocalUpdateLGFineTuning(BaseLocalUpdate):
    def __init__(self, args, user_idx=None, dataset=None, idxs=None, noise_logger=None, l_net=None):
        super().__init__(
            args=args,
            user_idx=user_idx,
            dataset=dataset,
            idxs=idxs,
            real_idx_return=True,
            noise_logger=noise_logger,
        )
        self.loss_func_no_re = nn.CrossEntropyLoss(reduce=False)
        self.loss_func = nn.CrossEntropyLoss()
        self.finetuning_train = DataLoader(
            DatasetSplit(dataset, idxs, idx_return=True),
            batch_size=self.args.local_bs,
            shuffle=True,
        )
        self.l_net = copy.deepcopy(l_net)

    def train(self, net, net2=None, gmm_infer=False):
        self.total_epochs += 1

        g_net = net
        l_net = copy.deepcopy(self.l_net)

        g_net.train()
        l_net.train()

        # train and update
        g_w = g_net.state_dict()
        l_w = l_net.state_dict()

        for key in g_w.keys():
            if 'linear' not in key:
                l_w[key] = copy.deepcopy(g_w[key])

        l_net.load_state_dict(l_w, strict=False)
        l_optimizer = torch.optim.SGD(l_net.parameters(), lr=self.args.lr, momentum=self.args.momentum,
                                      weight_decay=self.args.weight_decay)

        g_optimizer = torch.optim.SGD(g_net.parameters(), lr=self.args.lr, momentum=self.args.momentum,
                                      weight_decay=self.args.weight_decay)

        g_net.eval()

        # fine-tuning
        for iter in range(self.args.ft_local_ep):
            for batch_idx, (images, labels, indexes) in enumerate(self.finetuning_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                l_net.zero_grad()

                l_probs = l_net(images)
                #                 g_probs = g_net(images)

                #                 # Select clean data
                #                 g_loss = self.loss_func_no_re(g_probs, labels)

                #                 if gmm_infer:
                #                     pass
                #                 else:
                #                     ind_sorted = np.argsort(g_loss.data.cpu()).cuda()
                #                     loss_sorted = g_loss[ind_sorted]

                #                     remember_rate = 1 - self.args.forget_rate
                #                     num_remember = int(remember_rate * len(loss_sorted))

                #                     ind_update = ind_sorted[:num_remember]

                #                 loss_update = self.loss_func(l_probs[ind_update], labels[ind_update])

                #                 loss_l = torch.sum(loss_update) / num_remember
                #                 loss_l.backward()
                loss = self.loss_func(l_probs, labels)

                loss.backward()
                l_optimizer.step()

        l_net.eval()
        epoch_loss = []

        g_net.train()

        # Received global model training
        for local_ep in range(self.args.local_ep):
            batch_loss = []

            for batch_idx, batch in enumerate(self.ldr_train):
                images, labels, _, ids = batch
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                g_net.zero_grad()

                l_probs = l_net(images)
                g_probs = g_net(images)

                # Select clean data
                l_loss = self.loss_func_no_re(l_probs, labels)

                if gmm_infer:
                    pass
                else:
                    ind_sorted = np.argsort(l_loss.data.cpu()).cuda()
                    loss_sorted = l_loss[ind_sorted]

                    remember_rate = 1 - self.args.forget_rate
                    num_remember = int(remember_rate * len(loss_sorted))

                    ind_update = ind_sorted[:num_remember]

                loss_update = self.loss_func(g_probs[ind_update], labels[ind_update])

                loss_g = torch.sum(loss_update) / num_remember

                loss_g.backward()
                g_optimizer.step()

                batch_loss.append(loss_g.item())
                self.noise_logger.update(ids[ind_update])

            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            self.update_label_accuracy()

        return g_net.state_dict(), sum(epoch_loss) / len(epoch_loss)


#         loss_idx_lst = []
#         for batch_idx, (images, labels, indexes) in enumerate(self.finetuning_train):
#             images, labels = images.to(self.args.device), labels.to(self.args.device)
#             log_probs = l_net(images)
#             loss = self.loss_func_no_re(log_probs, labels)

#             for l, i in zip(loss, indexes):
#                 loss_idx_lst.append([l.item(), i.item()])

#         loss_idx_lst.sort(key=lambda x: x[0])

#         remember_rate = 1 - self.args.forget_rate
#         num_remember = int(remember_rate * len(loss_idx_lst))

#         filtered_train_data = []
#         dataset = self.finetuning_train.dataset

#         for l, i in loss_idx_lst[:num_remember]:
#             filtered_train_data.append((dataset[i][0], dataset[i][1]))

#         self.ldr_train = DataLoader(
#             filtered_train_data,
#             batch_size=int(self.args.local_bs * remember_rate),
#             shuffle=True,
#         )
#         return self.train_single_model(g_net)


class LocalUpdateFinetuning(BaseLocalUpdate):
    def __init__(self, args, user_idx=None, dataset=None, idxs=None, noise_logger=None):
        super().__init__(
            args=args,
            user_idx=user_idx,
            dataset=dataset,
            idxs=idxs,
            noise_logger=noise_logger,
        )
        self.loss_func_no_re = nn.CrossEntropyLoss(reduce=False)
        self.finetuning_train = DataLoader(
            DatasetSplit(dataset, idxs, idx_return=True),
            batch_size=self.args.local_bs,
            shuffle=True,
        )

    def train(self, net, net2=None):
        net.train()

        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)

        # Train and update
        g_w = copy.deepcopy(net.state_dict())

        # Fine-tuning
        for iter in range(self.args.ft_local_ep):
            for batch_idx, (images, labels, indexes) in enumerate(self.finetuning_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()

        # Select clean data
        net.eval()

        loss_idx_lst = []
        for batch_idx, (images, labels, indexes) in enumerate(self.finetuning_train):
            images, labels = images.to(self.args.device), labels.to(self.args.device)
            log_probs = net(images)
            loss = self.loss_func_no_re(log_probs, labels)

            for l, i in zip(loss, indexes):
                loss_idx_lst.append([l.item(), i.item()])

        loss_idx_lst.sort(key=lambda x: x[0])

        remember_rate = 1 - self.args.forget_rate
        num_remember = int(remember_rate * len(loss_idx_lst))

        filtered_train_data = []
        dataset = self.finetuning_train.dataset

        for l, i in loss_idx_lst[:num_remember]:
            filtered_train_data.append((dataset[i][0], dataset[i][1]))

        self.ldr_train = DataLoader(
            filtered_train_data,
            batch_size=int(self.args.local_bs * remember_rate),
            shuffle=True,
        )
        net.load_state_dict(g_w)
        return self.train_single_model(net)


class LocalUpdateGMix(BaseLocalUpdate):
    def __init__(self, args, user_idx=None, dataset=None, idxs=None, noise_logger=None):
        super().__init__(
            args=args,
            user_idx=user_idx,
            dataset=dataset,
            idxs=idxs,
            real_idx_return=True,
            noise_logger=noise_logger,
        )
        self.num_iter = int(len(idxs) / args.local_bs)
        self.semiloss = SemiLoss()

    def train(self, net, net2=None):
        if self.args.g_epoch <= self.args.warmup_epochs:
            return self.train_single_model(net)
        else:
            return self.train_2_phase(net)

    def train_2_phase(self, net):
        self.total_epochs += 1
        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)

        epoch_loss = []
        loss_func_no_re = nn.CrossEntropyLoss(reduce=False)

        net.eval()

        loss_idx_lst = []
        ids_list = []
        for batch_idx, (images, labels, indexes, ids) in enumerate(self.ldr_train):
            images, labels = images.to(self.args.device), labels.to(self.args.device)
            log_probs = net(images)
            loss = loss_func_no_re(log_probs, labels)

            for l, i in zip(loss, indexes):
                loss_idx_lst.append([l.item(), i.item()])
            ids_list.extend(ids)

        loss_idx_lst.sort(key=lambda x: x[0])

        remember_rate = 1 - self.args.forget_rate
        num_remember = int(remember_rate * len(loss_idx_lst))

        # Divide dataset into label & unlabel 
        labeled_idxs = []
        unlabeled_idxs = []

        for idx, (l, i) in enumerate(loss_idx_lst):
            if idx < num_remember:
                labeled_idxs.append(i)
            else:
                unlabeled_idxs.append(i)

        selected_ids = np.array(ids_list)[labeled_idxs]
        self.noise_logger.update(selected_ids)
        self.update_label_accuracy()

        #             for batch_idx, (images, labels, idxs) in enumerate(self.ldr_train):
        #                 images, labels = images.to(self.args.device), labels.to(self.args.device)
        #                 log_probs = net(images)

        #                 y_pred = log_probs.data.max(1, keepdim=True)[1]

        #                 for match, idx in zip(y_pred.eq(labels.data.view_as(y_pred)), idxs):
        #                     if match[0].item():
        #                         labeled_idxs.append(idx.item())
        #                     else:
        #                         unlabeled_idxs.append(idx.item())

        #             if len(labeled_idxs) % 2 != 0:
        #                 unlabeled_idxs.append(labeled_idxs.pop(-1))

        labeled_trainloader = DataLoader(PairDataset(self.dataset, labeled_idxs, label_return=True),
                                         batch_size=self.args.local_bs,
                                         shuffle=True)

        unlabeled_trainloader = DataLoader(PairDataset(self.dataset, unlabeled_idxs, label_return=False),
                                           batch_size=self.args.local_bs,
                                           shuffle=True)

        for ep in range(self.args.local_ep):
            labeled_train_iter = iter(labeled_trainloader)
            unlabeled_train_iter = iter(unlabeled_trainloader)

            net.train()

            batch_loss = []

            for batch_idx in range(self.num_iter):
                try:
                    inputs_x, inputs_x2, labels_x = labeled_train_iter.next()
                except:
                    labeled_train_iter = iter(labeled_trainloader)
                    inputs_x, inputs_x2, labels_x = labeled_train_iter.next()

                try:
                    inputs_u, inputs_u2 = unlabeled_train_iter.next()
                except:
                    unlabeled_train_iter = iter(unlabeled_trainloader)
                    inputs_u, inputs_u2 = unlabeled_train_iter.next()

                batch_size = inputs_x.size(0)

                labels_x = torch.zeros(batch_size, self.args.num_classes).scatter_(1, labels_x.view(-1, 1), 1)

                inputs_x, labels_x = inputs_x.to(self.args.device), labels_x.to(self.args.device)
                inputs_u, inputs_u2 = inputs_u.to(self.args.device), inputs_u2.to(self.args.device)

                with torch.no_grad():
                    outputs_u = net(inputs_u)
                    outputs_u2 = net(inputs_u2)
                    p = (torch.softmax(outputs_u, dim=1) + torch.softmax(outputs_u2, dim=1)) / 2
                    ptu = p ** (1 / self.args.T)
                    targets_u = ptu / ptu.sum(dim=1, keepdim=True)
                    targets_u = targets_u.detach()

                targets_x = labels_x
                all_inputs = torch.cat([inputs_x, inputs_u, inputs_u2], dim=0)
                all_targets = torch.cat([targets_x, targets_u, targets_u], dim=0)

                mixed_input, mixed_target = mixup(all_inputs, all_targets, alpha=self.args.mm_alpha)

                logits = net(mixed_input)
                logits_x = logits[:batch_size * 2]
                logits_u = logits[batch_size * 2:]

                loss = self.semiloss(
                    outputs_x=logits_x,
                    targets_x=mixed_target[:batch_size * 2],
                    targets_u=logits_u,
                    outputs_u=mixed_target[batch_size * 2:],
                    lambda_u=self.args.lambda_u,
                    epoch=self.args.g_epoch + batch_idx / self.num_iter,
                    warm_up=self.args.warmup_epochs,
                )
                # regularization
                prior = torch.ones(self.args.num_classes, device=self.args.device) / self.args.num_classes
                pred_mean = torch.softmax(logits, dim=1).mean(0)
                loss += torch.sum(prior * torch.log(prior / pred_mean))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if self.args.verbose and batch_idx % 10 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        ep, batch_idx * len(images), len(self.ldr_train.dataset),
                            100. * batch_idx / len(self.ldr_train), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)


class LocalUpdateDivideMix(BaseLocalUpdate):
    def __init__(self, args, user_idx=None, dataset=None, idxs=None, noise_logger=None, tmp_true_labels=None):
        super().__init__(
            args=args,
            user_idx=user_idx,
            dataset=dataset,
            idxs=idxs,
            idx_return=True,
            noise_logger=noise_logger,
            tmp_true_labels=tmp_true_labels
        )
        self.CE = nn.CrossEntropyLoss(reduction='none')
        self.CEloss = nn.CrossEntropyLoss()
        self.semiloss = SemiLoss()

        self.loss_history1 = []
        self.loss_history2 = []

        self.ldr_eval = DataLoader(
            DatasetSplit(dataset, idxs, real_idx_return=True),
            batch_size=self.args.local_bs,
            shuffle=False,
        )

    def train(self, client_num, net, net2=None):
        if self.args.g_epoch <= self.args.warmup_epochs:
            return self.train_multiple_models(client_num, net, net2)
        else:
            return self.train_2_phase(client_num, net, net2)

    def train_2_phase(self, client_num, net, net2):
        self.total_epochs += 1

        prob_dict1, label_idx1, unlabel_idx1 = self.update_probabilties_split_data_indices(net, self.loss_history1)
        prob_dict2, label_idx2, unlabel_idx2 = self.update_probabilties_split_data_indices(net2, self.loss_history2)

        self.noise_logger.update(label_idx1)
        self.update_label_accuracy()

        # train net1
        loss1 = self.divide_mix(
            net=net,
            net2=net2,
            label_idx=label_idx2,
            prob_dict=prob_dict2,
            unlabel_idx=unlabel_idx2,
            warm_up=self.args.warmup_epochs,
            epoch=self.args.g_epoch,
        )

        # train net2
        loss2 = self.divide_mix(
            net=net2,
            net2=net,
            label_idx=label_idx1,
            prob_dict=prob_dict1,
            unlabel_idx=unlabel_idx1,
            warm_up=self.args.warmup_epochs,
            epoch=self.args.g_epoch,
        )
        
        self.net1.load_state_dict(net.state_dict())   
        self.net2.load_state_dict(net2.state_dict())   
        
        if self.args.g_epoch in self.args.loss_dist_epoch:
            self.get_loss_dist(client_num=client_num, client=True)
        
        return net.state_dict(), loss1, net2.state_dict(), loss2

    def divide_mix(self, net, net2, label_idx, prob_dict, unlabel_idx, warm_up, epoch):
        net.train()
        net2.eval()  # fix one network and train the other

        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)

        # dataloader
        labeled_trainloader = DataLoader(
            PairProbDataset(self.dataset, label_idx, prob_dict),
            batch_size=self.args.local_bs,
            shuffle=True,
        )
        unlabeled_trainloader = DataLoader(
            PairDataset(self.dataset, unlabel_idx, label_return=False),
            batch_size=self.args.local_bs,
            shuffle=True,
        )
        unlabeled_train_iter = iter(unlabeled_trainloader)
        num_iter = len(labeled_trainloader)

        epoch_loss = []
        for ep in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (inputs_x, inputs_x2, labels_x, w_x) in enumerate(labeled_trainloader):
                try:
                    inputs_u, inputs_u2 = unlabeled_train_iter.next()
                except:
                    unlabeled_train_iter = iter(unlabeled_trainloader)
                    inputs_u, inputs_u2 = unlabeled_train_iter.next()

                batch_size = inputs_x.size(0)

                # Transform label to one-hot
                labels_x = torch.zeros(batch_size, self.args.num_classes) \
                    .scatter_(1, labels_x.view(-1, 1), 1)
                w_x = w_x.view(-1, 1).type(torch.FloatTensor)

                inputs_x = inputs_x.to(self.args.device)
                inputs_x2 = inputs_x2.to(self.args.device)
                labels_x = labels_x.to(self.args.device)
                w_x = w_x.to(self.args.device)

                inputs_u = inputs_u.to(self.args.device)
                inputs_u2 = inputs_u2.to(self.args.device)

                with torch.no_grad():
                    # label co-guessing of unlabeled samples
                    outputs_u11 = net(inputs_u)
                    outputs_u12 = net(inputs_u2)
                    outputs_u21 = net2(inputs_u)
                    outputs_u22 = net2(inputs_u2)

                    pu = (torch.softmax(outputs_u11, dim=1) + torch.softmax(outputs_u12, dim=1) +
                          torch.softmax(outputs_u21, dim=1) + torch.softmax(outputs_u22, dim=1)) / 4
                    ptu = pu ** (1 / self.args.T)  # temparature sharpening

                    targets_u = ptu / ptu.sum(dim=1, keepdim=True)  # normalize
                    targets_u = targets_u.detach()

                    # label refinement of labeled samples
                    outputs_x = net(inputs_x)
                    outputs_x2 = net(inputs_x2)

                    px = (torch.softmax(outputs_x, dim=1) + torch.softmax(outputs_x2, dim=1)) / 2
                    px = w_x * labels_x + (1 - w_x) * px
                    ptx = px ** (1 / self.args.T)  # temparature sharpening

                    targets_x = ptx / ptx.sum(dim=1, keepdim=True)  # normalize
                    targets_x = targets_x.detach()

                # mixmatch
                all_inputs = torch.cat([inputs_x, inputs_x2, inputs_u, inputs_u2], dim=0)
                all_targets = torch.cat([targets_x, targets_x, targets_u, targets_u], dim=0)

                mixed_input, mixed_target = mixup(all_inputs, all_targets, alpha=self.args.mm_alpha)

                logits = net(mixed_input)
                # compute loss
                loss = self.semiloss(
                    outputs_x=logits[:batch_size * 2],
                    targets_x=mixed_target[:batch_size * 2],
                    outputs_u=logits[batch_size * 2:],
                    targets_u=mixed_target[batch_size * 2:],
                    lambda_u=self.args.lambda_u,
                    epoch=epoch + batch_idx / num_iter,
                    warm_up=warm_up,
                )
                # regularization
                prior = torch.ones(self.args.num_classes, device=self.args.device) / self.args.num_classes
                pred_mean = torch.softmax(logits, dim=1).mean(0)
                penalty = torch.sum(prior * torch.log(prior / pred_mean))
                loss += penalty

                # compute gradient and do SGD step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                batch_loss.append(loss.item())

            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        return sum(epoch_loss) / len(epoch_loss)

    def update_probabilties_split_data_indices(self, model, loss_history):
        model.eval()
        losses_lst = []
        idx_lst = []

        with torch.no_grad():
            for batch_idx, (inputs, targets, items, idxs) in enumerate(self.ldr_eval):
                inputs, targets = inputs.to(self.args.device), targets.to(self.args.device)
                outputs = model(inputs)
                losses_lst.append(self.CE(outputs, targets))
                idx_lst.append(idxs.cpu().numpy())

        indices = np.concatenate(idx_lst)
        losses = torch.cat(losses_lst).cpu().numpy()
        losses = (losses - losses.min()) / (losses.max() - losses.min())
        loss_history.append(losses)

        # Fit a two-component GMM to the loss
        input_loss = losses.reshape(-1, 1)
        gmm = GaussianMixture(n_components=2, max_iter=100, tol=1e-2, reg_covar=5e-4)
        gmm.fit(input_loss)
        prob = gmm.predict_proba(input_loss)
        prob = prob[:, gmm.means_.argmin()]

        # Split data to labeled, unlabeled dataset
        pred = (prob > self.args.p_threshold)
        label_idx = pred.nonzero()[0]
        label_idx = indices[label_idx]

        unlabel_idx = (1 - pred).nonzero()[0]
        unlabel_idx = indices[unlabel_idx]

        # Data index : probability
        prob_dict = {idx: prob for idx, prob in zip(indices, prob)}

        return prob_dict, label_idx, unlabel_idx


class LocalUpdateHS(BaseLocalUpdate):
    def __init__(self, args, user_idx=None, dataset=None, idxs=None, noise_logger=None):
        super().__init__(
            args=args,
            user_idx=user_idx,
            dataset=dataset,
            idxs=idxs,
            noise_logger=noise_logger,
        )
        self.loss_func_no_red = nn.CrossEntropyLoss(reduction="none")
        self.count = 0
        self.ldr_train = DataLoader(
            DatasetSplitHS(self.dataset, self.idxs),
            batch_size=self.args.local_bs,
            shuffle=True,
        )

    def train(self, net, net2=None):
        self.count += 1
        if self.args.g_epoch <= self.args.warmup_epochs:
            return self.train_single_model(net)
        else:
            return self.train_2_phase(net)

    def forward_pass(self, batch, net, net2=None):
        images, labels, soft_labels, predictions, indexes = batch
        images, labels, soft_labels = images.to(self.args.device), labels.to(self.args.device), soft_labels.to(
            self.args.device)

        log_probs = net(images)
        loss = self.loss_func(log_probs, labels)

        if self.batch_idx == 0:
            for prob, index in zip(log_probs, indexes):
                self.dataset.prediction[index][self.count % self.args.queue_size] = F.softmax(
                    prob.cpu()).detach().numpy()
        return loss

    def train_2_phase(self, net):
        # Select clean data
        net.eval()

        ldr_train = DataLoader(
            DatasetSplitHS(self.dataset, self.idxs),
            batch_size=self.args.local_bs,
            shuffle=True,
        )
        loss_idx_lst = []
        for batch_idx, (images, labels, soft_labels, pred_lst, indexes) in enumerate(ldr_train):
            images, labels = images.to(self.args.device), labels.to(self.args.device)
            log_probs = net(images)
            loss = self.loss_func_no_red(log_probs, labels)

            for l, i, prob in zip(loss, indexes, log_probs):
                self.dataset.prediction[i][self.count % self.args.queue_size] = prob.cpu().detach().numpy()
                loss_idx_lst.append([l.item(), i.item()])

        loss_idx_lst.sort(key=lambda x: x[0])

        remember_rate = 1 - self.args.forget_rate
        num_remember = int(remember_rate * len(loss_idx_lst))

        # pseudo label (Hard Label)
        for l, i in loss_idx_lst[num_remember:]:
            self.dataset.train_labels[i] = int(np.argmax(np.mean(self.dataset.prediction[i], axis=0)))

        self.ldr_train = DataLoader(
            DatasetSplitHS(self.dataset, self.idxs),
            batch_size=self.args.local_bs,
            shuffle=True,
        )
        return self.train_single_model(net)

    def criterion_softlabel(self, outputs, soft_targets):
        return self.loss_func(F.softmax(outputs), soft_targets)


class LocalUpdateLGCorrection(LocalUpdateSELFIE):
    def __init__(self, args, user_idx=None, dataset=None, idxs=None, noise_rate=0, noise_logger=None,
                 tmp_true_labels=None, l_net=None):
        super().__init__(
            args=args,
            user_idx=user_idx,
            dataset=dataset,
            idxs=idxs,
            noise_rate=noise_rate,
            noise_logger=noise_logger,
            tmp_true_labels=tmp_true_labels
        )

        self.l_net = l_net

    def train(self, net):
        g_net = copy.deepcopy(net)

        optimizer = torch.optim.SGD(
            net.parameters(),
            lr=self.args.lr,
            momentum=self.args.momentum,
            weight_decay=self.args.weight_decay,
        )

        epoch_loss = []
        for epoch in range(self.args.local_ep):
            self.epoch = epoch
            batch_loss = []
            for batch_idx, batch in enumerate(self.ldr_train):
                self.batch_idx = batch_idx

                net.zero_grad()
                images, labels, _, ids = batch
                images = images.to(self.args.device)
                labels = labels.to(self.args.device)
                ids = ids.numpy()

                log_probs = net(images)
                loss_array = self.loss_func(log_probs, labels)

                # update prediction history
                self.corrector.update_prediction_history(
                    ids=ids,
                    outputs=log_probs.cpu().detach().numpy(),
                )

                if self.total_epochs >= self.warmup:
                    # correct labels, remove noisy data
                    images, labels = self.corrector.patch_clean_with_corrected_sample_batch(
                        ids=ids,
                        X=images,
                        y=labels,
                        loss_array=loss_array.cpu().detach().numpy(),
                    )

                    images = images.to(self.args.device)
                    labels = labels.to(self.args.device)
                    log_probs = net(images)
                    loss_array = self.loss_func(log_probs, labels)

                loss = loss_array.mean()
                fed_prox_reg = 0.0

                for l_param, g_param in zip(net.parameters(), g_net.parameters()):
                    fed_prox_reg += (0.01 / 2 * torch.norm((l_param - g_param)) ** 2)
                loss += fed_prox_reg

                loss.backward()
                optimizer.step()

                batch_loss.append(loss.item())
                self.on_batch_end()

            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            self.total_epochs += 1
            self.on_epoch_end()

        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)
