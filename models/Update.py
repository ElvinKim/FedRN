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


# from utils.logger import get_loss_dist


class NegEntropy(object):
    def __call__(self, outputs):
        probs = torch.softmax(outputs, dim=1)
        return torch.mean(torch.sum(probs.log() * probs, dim=1))


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
                             noise_logger=None, tmp_true_labels=None, gaussian_noise=None):
    local_update_objects = []
    for idx, noise_rate in zip(range(args.num_users), noise_rates):
        local_update_args = dict(
            args=args,
            user_idx=idx,
            dataset=dataset_train,
            idxs=dict_users[idx],
            noise_logger=noise_logger,
            tmp_true_labels=tmp_true_labels,
        )

        if args.method == 'default':
            local_update_object = BaseLocalUpdate(**local_update_args, gaussian_noise=gaussian_noise)

        elif args.method == 'ours':
            local_update_object = LocalUpdateOurs(**local_update_args, gaussian_noise=gaussian_noise)

        elif args.method == 'selfie':
            local_update_object = LocalUpdateSELFIE(noise_rate=noise_rate, **local_update_args)

        elif args.method == 'jointoptim':
            local_update_object = LocalUpdateJointOptim(**local_update_args)

        elif args.method in ['coteaching', 'coteaching+']:
            local_update_object = LocalUpdateCoteaching(is_coteaching_plus=bool(args.method == 'coteaching+'),
                                                        **local_update_args)
        elif args.method == 'dividemix':
            local_update_object = LocalUpdateDivideMix(**local_update_args)

        elif args.method == 'fedprox':
            local_update_object = LocalUpdateFedProx(**local_update_args)

        elif args.method == 'RFL':
            local_update_object = LocalUpdateRFL(**local_update_args)

        elif args.method == 'global_model':
            local_update_object = LocalUpdateGlobalModel(**local_update_args)

        elif args.method == 'global_GMM_base':
            local_update_object = LocalUpdateGlobalGMMBase(**local_update_args)

        elif args.method == 'global_with_neighbors':
            local_update_object = LocalUpdateGlobalWithNeighbors(**local_update_args)

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
            tmp_true_labels=None,
            gaussian_noise=None
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
            num_workers=self.args.num_workers,
            pin_memory=True,
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

        self.last_updated = 0

        self.gaussian_noise = gaussian_noise
        self.conf_penalty = False

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
            for batch_idx, batch in enumerate(
                    DataLoader(dataset, batch_size=1, shuffle=False, num_workers=self.args.num_workers, pin_memory=True)
            ):
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

    def forward_pass(self, batch, net, net2=None, conf_penalty=None):
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
        if conf_penalty:
            penalty = NegEntropy()
            loss += penalty(log_probs)
        if net2 is None:
            return loss

        # 2 models
        log_probs2 = net2(images)
        loss2 = self.loss_func(log_probs2, labels)
        return loss, loss2

    def on_batch_end(self):
        pass

    def on_epoch_end(self):
        if self.args.method != "default":
            self.update_label_accuracy()


class LocalUpdateOurs(BaseLocalUpdate):
    def __init__(self, args, dataset=None, user_idx=None, idxs=None, noise_logger=None, tmp_true_labels=None,
                 gaussian_noise=None):
        super().__init__(
            args=args,
            dataset=dataset,
            user_idx=user_idx,
            idxs=idxs,
            real_idx_return=True,
            noise_logger=noise_logger,
            gaussian_noise=gaussian_noise
        )
        self.CE = nn.CrossEntropyLoss(reduction='none')

        self.ldr_eval = DataLoader(
            DatasetSplit(dataset, idxs, real_idx_return=True),
            batch_size=self.args.local_bs,
            shuffle=True,
            num_workers=self.args.num_workers,
            pin_memory=True,
        )

        self.conf_penalty = False
        self.expertise = 0.5
        self.arbitrary_output = torch.rand((1, 10))

    def split_data_indices(self, prev, neighbor_lst, neighbor_score_lst):
        prev.eval()

        for n_net in neighbor_lst:
            n_net.eval()

        losses_lst = []
        idx_lst = []

        # get weighted sum of losses
        with torch.no_grad():
            for batch_idx, (inputs, targets, items, idxs) in enumerate(self.ldr_eval):
                inputs, targets = inputs.to(self.args.device), targets.to(self.args.device)
                outputs = prev(inputs)
                loss = self.CE(outputs, targets)

                for n_net, score in zip(neighbor_lst, neighbor_score_lst):
                    n_outputs = n_net(inputs)
                    loss += self.CE(n_outputs, targets) * score

                losses_lst.append(loss)
                idx_lst.append(idxs.cpu().numpy())

        indices = np.concatenate(idx_lst)
        losses = torch.cat(losses_lst).cpu().numpy()
        losses = (losses - losses.min()) / (losses.max() - losses.min())

        # Fit a two-component GMM to the loss
        input_loss = losses.reshape(-1, 1)
        gmm = GaussianMixture(n_components=2, max_iter=100, tol=1e-2, reg_covar=5e-4)
        gmm.fit(input_loss)
        prob = gmm.predict_proba(input_loss)
        prob = prob[:, gmm.means_.argmin()]

        # Split data to clean, noisy dataset
        threshold = 0.5
        pred = (prob > threshold)

        clean_idx = pred.nonzero()[0]
        clean_idx = indices[clean_idx]

        noisy_idx = (1 - pred).nonzero()[0]
        noisy_idx = indices[noisy_idx]

        return clean_idx, noisy_idx

    def finetune(self, neighbor_lst):
        optimizer_args = dict(
            lr=self.args.lr,
            momentum=self.args.momentum,
            weight_decay=self.args.weight_decay,
        )

        n_opt_lst = []

        for n_net in neighbor_lst:
            n_net.train()
            n_opt_lst.append(torch.optim.SGD(n_net.parameters(), **optimizer_args))

        for batch_idx, (inputs, targets, items, idxs) in enumerate(self.ldr_eval):
            inputs, targets = inputs.to(self.args.device), targets.to(self.args.device)

            for n_net, n_opt in zip(neighbor_lst, n_opt_lst):
                n_net.zero_grad()

                outputs = n_net(inputs)
                loss = self.loss_func(outputs, targets)
                loss.backward()

                n_opt.step()

        return neighbor_lst

    def train_phase1(self, client_num, net):
        # local training
        net.train()
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

                loss = self.forward_pass(batch, net, conf_penalty=self.conf_penalty)
                loss.backward()
                optimizer.step()

                batch_loss.append(loss.item())
                self.on_batch_end()

            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            self.total_epochs += 1
            self.on_epoch_end()

        self.net1.load_state_dict(net.state_dict())

        self.last_updated = self.args.g_epoch

        if self.args.g_epoch in self.args.loss_dist_epoch:
            self.get_loss_dist(client_num=client_num, client=True)

        net.eval()
        correct = 0
        n_total = len(self.ldr_eval.dataset)

        # get expertise of the client
        with torch.no_grad():
            for batch_idx, (inputs, targets, items, idxs) in enumerate(self.ldr_eval):
                inputs, targets = inputs.to(self.args.device), targets.to(self.args.device)
                outputs = self.net1(inputs)
                y_pred = outputs.data.max(1, keepdim=True)[1]
                correct += y_pred.eq(targets.data.view_as(y_pred)).float().sum().item()
            expertise = correct / n_total

        self.expertise = expertise

        # arbitrary gaussian input inference
        self.arbitrary_output = net(self.gaussian_noise.to(self.args.device))

        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def train_phase2(self, client_num, net, neighbor_lst, neighbor_score_lst):
        neigbor_lst = self.finetune(neighbor_lst)

        # fit GMM & get clean data index
        clean_idx, noisy_idx = self.split_data_indices(self.net1, neighbor_lst, neighbor_score_lst)

        self.ldr_train = DataLoader(DatasetSplit(self.dataset, clean_idx, real_idx_return=True),
                                    batch_size=self.args.local_bs,
                                    shuffle=True,
                                    num_workers=self.args.num_workers,
                                    pin_memory=True,
                                    )

        # local training
        net.train()
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

                batch_loss.append(loss.item())
                self.on_batch_end()

            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            self.total_epochs += 1
            self.on_epoch_end()

        self.net1.load_state_dict(net.state_dict())

        self.last_updated = self.args.g_epoch

        if self.args.g_epoch in self.args.loss_dist_epoch:
            self.get_loss_dist(client_num=client_num, client=True)

        net.eval()
        correct = 0
        n_total = len(self.ldr_eval.dataset)

        # get expertise of the client
        with torch.no_grad():
            for batch_idx, (inputs, targets, items, idxs) in enumerate(self.ldr_eval):
                inputs, targets = inputs.to(self.args.device), targets.to(self.args.device)
                outputs = self.net1(inputs)
                y_pred = outputs.data.max(1, keepdim=True)[1]
                correct += y_pred.eq(targets.data.view_as(y_pred)).float().sum().item()
            expertise = correct / n_total

        self.expertise = expertise

        # arbitrary gaussian input inference
        self.arbitrary_output = net(self.gaussian_noise.to(self.args.device))

        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)


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
            num_workers=self.args.num_workers,
            pin_memory=True,
        )
        self.ldr_train_tmp = DataLoader(
            DatasetSplitRFL(dataset, idxs),
            batch_size=1,
            shuffle=True,
            num_workers=self.args.num_workers,
            pin_memory=True,
        )
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

    def forward_pass(self, batch, net, net2=None, conf_penalty=None):
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

    def forward_pass(self, batch, net, net2=None, conf_penalty=None):
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

    def forward_pass(self, batch, net, net2=None, conf_penalty=None):
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

    def forward_pass(self, batch, net, net2=None, conf_penalty=None):
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
            num_workers=self.args.num_workers,
            pin_memory=True,
        )

    def train(self, client_num, net, net2=None):
        if self.args.g_epoch <= self.args.warmup_epochs:
            return self.train_multiple_models(client_num, net, net2)
        else:
            return self.train_2_phase(client_num, net, net2)

    def train_2_phase(self, client_num, net, net2):
        epoch_loss1 = []
        epoch_loss2 = []

        for ep in range(self.args.local_ep):
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

            self.total_epochs += 1
            epoch_loss1.append(loss1)
            epoch_loss2.append(loss2)

        loss1 = sum(epoch_loss1) / len(epoch_loss1)
        loss2 = sum(epoch_loss2) / len(epoch_loss2)
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
            num_workers=self.args.num_workers,
            pin_memory=True,
        )
        unlabeled_trainloader = DataLoader(
            PairDataset(self.dataset, unlabel_idx, label_return=False),
            batch_size=self.args.local_bs,
            shuffle=True,
            num_workers=self.args.num_workers,
            pin_memory=True,
        )
        unlabeled_train_iter = iter(unlabeled_trainloader)
        num_iter = len(labeled_trainloader)

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

        return sum(batch_loss) / len(batch_loss)

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


class LocalUpdateGlobalModel(BaseLocalUpdate):
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

        self.CE = nn.CrossEntropyLoss(reduction='none')

    def forward_pass(self, batch, net, net2=None, conf_penalty=None):
        images, labels, indices, ids = batch

        images = images.to(self.args.device)
        labels = labels.to(self.args.device)
        log_probs = net(images)

        loss = self.CE(log_probs, labels)
        ind_sorted = torch.argsort(loss)

        remember_rate = 1 - self.args.forget_rate
        num_remember = int(remember_rate * len(ind_sorted))

        ind_update = ind_sorted[:num_remember]

        loss_update = self.CE(log_probs[ind_update], labels[ind_update])

        self.noise_logger.evaluated_data = ids.tolist()
        self.noise_logger.update(ids[ind_update])

        self.update_label_accuracy()

        return torch.sum(loss_update) / num_remember

    def on_epoch_end(self):
        pass


class LocalUpdateGlobalGMMBase(BaseLocalUpdate):
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

        self.CE = nn.CrossEntropyLoss(reduction='none')

        self.ldr_eval = DataLoader(
            DatasetSplit(dataset, idxs, real_idx_return=True),
            batch_size=self.args.local_bs,
            shuffle=True,
            num_workers=self.args.num_workers,
            pin_memory=True,
        )

    def train(self, client_num, net):

        net.train()

        optimizer = torch.optim.SGD(
            net.parameters(),
            lr=self.args.lr,
            momentum=self.args.momentum,
            weight_decay=self.args.weight_decay,
        )

        epoch_loss = []

        for epoch in range(self.args.local_ep):

            if self.args.g_epoch > self.args.warmup_epochs:
                label_idx, unlabel_idx = self.update_probabilties_split_data_indices(net)

                self.ldr_train = DataLoader(
                    DatasetSplit(self.dataset, label_idx, real_idx_return=True),
                    batch_size=self.args.local_bs,
                    shuffle=True,
                    num_workers=self.args.num_workers,
                    pin_memory=True,
                )

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

    def update_probabilties_split_data_indices(self, model):
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

        return label_idx, unlabel_idx


class LocalUpdateGlobalWithNeighbors(LocalUpdateGlobalGMMBase):
    def __init__(self, args, user_idx=None, dataset=None, idxs=None, noise_logger=None, tmp_true_labels=None):
        super().__init__(
            args=args,
            user_idx=user_idx,
            dataset=dataset,
            idxs=idxs,
            noise_logger=noise_logger,
            tmp_true_labels=tmp_true_labels
        )

        self.CE = nn.CrossEntropyLoss(reduction='none')

        self.ldr_eval = DataLoader(
            DatasetSplit(dataset, idxs, real_idx_return=True),
            batch_size=self.args.local_bs,
            shuffle=True,
            num_workers=self.args.num_workers,
            pin_memory=True,
        )

    def train(self, client_num, net, neighbor_lst):

        net.train()

        optimizer = torch.optim.SGD(
            net.parameters(),
            lr=self.args.lr,
            momentum=self.args.momentum,
            weight_decay=self.args.weight_decay,
        )

        ''' Method 5
        neighbor_opt_lst = []
        
        if self.args.g_epoch > self.args.warmup_epochs:
            for n_net in neighbor_lst:
                n_opt = torch.optim.SGD(
                    n_net.parameters(),
                    lr=self.args.lr,
                    momentum=self.args.momentum,
                    weight_decay=self.args.weight_decay,
                    )
                neighbor_opt_lst.append(n_opt) 
        
        '''

        epoch_loss = []

        if self.args.g_epoch > self.args.warmup_epochs:
            neighbor_opt_lst = []

            for n_net in neighbor_lst:
                n_opt = torch.optim.SGD(
                    n_net.parameters(),
                    lr=self.args.lr,
                    momentum=self.args.momentum,
                    weight_decay=self.args.weight_decay,
                )
                neighbor_opt_lst.append(n_opt)

            temp_data_loader = DataLoader(
                DatasetSplit(self.dataset, self.idxs, real_idx_return=True),
                batch_size=self.args.local_bs,
                shuffle=True,
                num_workers=self.args.num_workers,
                pin_memory=True,
            )

            for batch_idx, batch in enumerate(temp_data_loader):
                self.batch_idx = batch_idx
                net.zero_grad()

                for n_net, n_net_opt in zip(neighbor_lst, neighbor_opt_lst):
                    n_net.zero_grad()
                    loss = self.forward_pass(batch, n_net)
                    loss.backward()
                    n_net_opt.step()

        for epoch in range(self.args.local_ep):
            if self.args.g_epoch > self.args.warmup_epochs:
                '''
                Method 4 
                idxs = []
                temp_data_loader = DataLoader(
                    DatasetSplit(self.dataset, self.idxs, real_idx_return=True),
                    batch_size=1,
                    shuffle=True,
                )
                
                for data, target, item, idx in temp_data_loader:
                    data_check = []
                    data, target = data.to(self.args.device), target.to(self.args.device)
                    
                    log_prob = net(data)
                    
                    # get the index of the max log-probability
                    y_pred = log_prob.data.max(1, keepdim=True)[1]
                    
                    data_check.append(y_pred[0].item() == target.item())
                    
                    for n_net in neighbor_lst:
                        log_prob = n_net(data)

                        # get the index of the max log-probability
                        y_pred = log_prob.data.max(1, keepdim=True)[1]

                        data_check.append(y_pred[0].item() == target.item())
                        
                    if any(data_check):
                        idxs.append(idx.item())
                        
                self.ldr_eval = DataLoader(
                    DatasetSplit(self.dataset, idxs, real_idx_return=True),
                    batch_size=self.args.local_bs,
                    shuffle=True,
                )
                '''

                ''' Method 1
                label_idx, unlabel_idx = self.update_probabilties_split_data_indices(net, neighbor_lst) 
                '''

                label_idx, unlabel_idx = self.update_probabilties_split_data_indices(net)

                '''Method 3
                neighbor_label_idx = []
                for neighbor_net in neighbor_lst:
                    temp_label_idx, temp_unlabel_idx = self.update_probabilties_split_data_indices(neighbor_net)
                    neighbor_label_idx = list(set(neighbor_label_idx) | set(temp_label_idx))
                label_idx = list(set(neighbor_label_idx) & set(label_idx))
                
                '''

                ''' Method 7
                neighbor_label_idx = []
                intersec_label_idx = []
                for neighbor_net in neighbor_lst:
                    temp_label_idx, temp_unlabel_idx = self.update_probabilties_split_data_indices(neighbor_net)
                    neighbor_label_idx = list(set(neighbor_label_idx) | set(temp_label_idx))
                    
                    if len(intersec_label_idx) == 0:
                        intersec_label_idx = temp_label_idx
                    else:
                        intersec_label_idx = list(set(temp_label_idx) & set(intersec_label_idx))
                        
                label_idx = list(set(neighbor_label_idx) & set(label_idx)) + intersec_label_idx
                '''
                label_idx = list(label_idx)
                unlabel_idx = list(unlabel_idx)

                neighbor_label_idx = []
                for neighbor_net in neighbor_lst:
                    temp_label_idx, temp_unlabel_idx = self.update_probabilties_split_data_indices(neighbor_net)
                    neighbor_label_idx = list(set(neighbor_label_idx) | set(temp_label_idx))

                temp_label_idx = list(set(neighbor_label_idx) & set(label_idx))

                #                 unlabel_idx += list(set(label_idx) - set(temp_label_idx))

                label_idx = temp_label_idx

                #                 # Move data from predicted noisy to predicted clean
                #                 idxs = []
                #                 temp_data_loader = DataLoader(
                #                     DatasetSplit(self.dataset, unlabel_idx, real_idx_return=True),
                #                     batch_size=1,
                #                     shuffle=True,
                #                 )

                #                 for data, target, item, idx in temp_data_loader:
                #                     data_check = []
                #                     data, target = data.to(self.args.device), target.to(self.args.device)

                #                     log_prob = net(data)

                #                     # get the index of the max log-probability
                #                     y_pred = log_prob.data.max(1, keepdim=True)[1]

                #                     data_check.append(y_pred[0].item() == target.item())

                #                     for n_net in neighbor_lst:
                #                         log_prob = n_net(data)

                #                         # get the index of the max log-probability
                #                         y_pred = log_prob.data.max(1, keepdim=True)[1]

                #                         data_check.append(y_pred[0].item() == target.item())

                #                     if all(data_check):
                #                         idxs.append(idx.item())
                #                         label_idx.append(idx.item())
                #                         unlabel_idx.remove(idx.item())
                #                 print("move noisy to clean", len(idxs))

                self.ldr_train = DataLoader(
                    DatasetSplit(self.dataset, label_idx, real_idx_return=True),
                    batch_size=self.args.local_bs,
                    shuffle=True,
                    num_workers=self.args.num_workers,
                    pin_memory=True,
                )

            self.epoch = epoch
            batch_loss = []
            for batch_idx, batch in enumerate(self.ldr_train):
                self.batch_idx = batch_idx
                net.zero_grad()

                loss = self.forward_pass(batch, net)
                loss.backward()
                optimizer.step()

                batch_loss.append(loss.item())

                ''' Method 5
                if self.args.g_epoch > self.args.warmup_epochs:
                    for n_net, n_opt in zip(neighbor_lst, neighbor_opt_lst):
                        net.zero_grad()
                        
                        n_loss = self.forward_pass(batch, n_net)
                        n_loss.backward()
                        n_opt.step()
                '''

            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            self.total_epochs += 1
            self.on_epoch_end()

        self.net1.load_state_dict(net.state_dict())

        if self.args.g_epoch in self.args.loss_dist_epoch:
            self.get_loss_dist(client_num=client_num, client=True)

        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)

    #     def update_probabilties_split_data_indices(self, model, neighbor_lst):
    #         model.eval()
    #         losses_lst = []
    #         idx_lst = []

    #         with torch.no_grad():
    #             for batch_idx, (inputs, targets, items, idxs) in enumerate(self.ldr_eval):
    #                 inputs, targets = inputs.to(self.args.device), targets.to(self.args.device)
    #                 outputs = model(inputs)
    #                 losses = self.CE(outputs, targets)

    #                 for neighbor in neighbor_lst:
    #                     neighbor.eval()
    #                     outputs = neighbor(inputs)
    #                     losses += self.CE(outputs, targets)

    #                 losses_lst.append(losses)
    #                 idx_lst.append(idxs.cpu().numpy())

    #         indices = np.concatenate(idx_lst)
    #         losses = torch.cat(losses_lst).cpu().numpy()
    #         losses = (losses - losses.min()) / (losses.max() - losses.min())

    #         # Fit a two-component GMM to the loss
    #         input_loss = losses.reshape(-1, 1)
    #         gmm = GaussianMixture(n_components=2, max_iter=100, tol=1e-2, reg_covar=5e-4)
    #         gmm.fit(input_loss)
    #         prob = gmm.predict_proba(input_loss)
    #         prob = prob[:, gmm.means_.argmin()]

    #         # Split data to labeled, unlabeled dataset
    #         pred = (prob > self.args.p_threshold)
    #         label_idx = pred.nonzero()[0]
    #         label_idx = indices[label_idx]

    #         unlabel_idx = (1 - pred).nonzero()[0]
    #         unlabel_idx = indices[unlabel_idx]

    #         return label_idx, unlabel_idx

    def update_probabilties_split_data_indices(self, model):
        model.eval()
        losses_lst = []
        idx_lst = []

        with torch.no_grad():
            for batch_idx, (inputs, targets, items, idxs) in enumerate(self.ldr_eval):
                inputs, targets = inputs.to(self.args.device), targets.to(self.args.device)
                outputs = model(inputs)
                losses = self.CE(outputs, targets)

                losses_lst.append(losses)
                idx_lst.append(idxs.cpu().numpy())

        indices = np.concatenate(idx_lst)
        losses = torch.cat(losses_lst).cpu().numpy()
        losses = (losses - losses.min()) / (losses.max() - losses.min())

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

        return label_idx, unlabel_idx
