#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import random
from sklearn import metrics
import copy
from sklearn.mixture import GaussianMixture
from .selfie_corrector import SelfieCorrector


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


class MixMatchLabeled(Dataset):
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


class MixMatchUnlabeled(Dataset):
    def __init__(self, dataset, idxs, idx_return=False):
        self.dataset = dataset
        self.idxs = list(idxs)
        self.idx_return = idx_return

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        item = int(item)
        image1, label = self.dataset[self.idxs[item]]
        image2, label = self.dataset[self.idxs[item]]

        if self.idx_return:
            return image1, image2, item
        else:
            return image1, image2


class SemiLoss(object):
    def __call__(self, args, outputs_x, targets_x, outputs_u, targets_u, epoch, warm_up):
        probs_u = torch.softmax(outputs_u, dim=1)

        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        Lu = torch.mean((probs_u - targets_u) ** 2)

        return Lx, Lu, linear_rampup(args, epoch, warm_up)


def linear_rampup(args, current, warm_up, rampup_length=16):
    current = np.clip((current - warm_up) / rampup_length, 0.0, 1.0)
    return args.lambda_u * float(current)


class GMixLabeled(Dataset):
    def __init__(self, dataset, idxs, idx_return=False):
        self.dataset = dataset
        self.idxs = list(idxs)
        self.idx_return = idx_return

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        item = int(item)
        image1, label = self.dataset[self.idxs[item]]
        image2, label = self.dataset[self.idxs[item]]

        if self.idx_return:
            return image1, image2, label, item
        else:
            return image1, image2, label


class GMixUnlabeled(Dataset):
    def __init__(self, dataset, idxs, idx_return=False):
        self.dataset = dataset
        self.idxs = list(idxs)
        self.idx_return = idx_return

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        item = int(item)
        image1, label = self.dataset[self.idxs[item]]
        image2, label = self.dataset[self.idxs[item]]

        if self.idx_return:
            return image1, image2, item
        else:
            return image1, image2


def loss_coteaching(y_pred1, y_pred2, y_true, forget_rate, loss_func=None):
    loss_1 = loss_func(y_pred1, y_true)
    ind_1_sorted = np.argsort(loss_1.data.cpu()).cuda()
    loss_1_sorted = loss_1[ind_1_sorted]

    loss_2 = loss_func(y_pred2, y_true)
    ind_2_sorted = np.argsort(loss_2.data.cpu()).cuda()
    loss_2_sorted = loss_2[ind_2_sorted]

    remember_rate = 1 - forget_rate
    num_remember = int(remember_rate * len(loss_1_sorted))

    ind_1_update = ind_1_sorted[:num_remember]
    ind_2_update = ind_2_sorted[:num_remember]
    # exchange
    loss_1_update = loss_func(y_pred1[ind_2_update], y_true[ind_2_update])
    loss_2_update = loss_func(y_pred2[ind_1_update], y_true[ind_1_update])

    return torch.sum(loss_1_update) / num_remember, torch.sum(loss_2_update) / num_remember


def loss_coteaching_plus(logits, logits2, labels, forget_rate, ind, step, loss_func=None):
    outputs = F.softmax(logits, dim=1)
    outputs2 = F.softmax(logits2, dim=1)

    _, pred1 = torch.max(logits.data, 1)
    _, pred2 = torch.max(logits2.data, 1)

    pred1, pred2 = pred1.cpu().numpy(), pred2.cpu().numpy()

    logical_disagree_id = np.zeros(labels.size(), dtype=bool)
    disagree_id = []
    for idx, p1 in enumerate(pred1):
        if p1 != pred2[idx]:
            disagree_id.append(idx)
            logical_disagree_id[idx] = True

    temp_disagree = ind * logical_disagree_id.astype(np.int64)
    ind_disagree = np.asarray([i for i in temp_disagree if i != 0]).transpose()
    try:
        assert ind_disagree.shape[0] == len(disagree_id)
    except:
        disagree_id = disagree_id[:ind_disagree.shape[0]]

    _update_step = np.logical_or(logical_disagree_id, step < 5000).astype(np.float32)
    update_step = Variable(torch.from_numpy(_update_step)).cuda()

    if len(disagree_id) > 0:
        update_labels = labels[disagree_id]
        update_outputs = outputs[disagree_id]
        update_outputs2 = outputs2[disagree_id]

        loss_1, loss_2, = loss_coteaching(update_outputs, update_outputs2, update_labels, forget_rate, loss_func)
    else:
        update_labels = labels
        update_outputs = outputs
        update_outputs2 = outputs2

        cross_entropy_1 = F.cross_entropy(update_outputs, update_labels)
        cross_entropy_2 = F.cross_entropy(update_outputs2, update_labels)

        loss_1 = torch.sum(update_step * cross_entropy_1) / labels.size()[0]
        loss_2 = torch.sum(update_step * cross_entropy_2) / labels.size()[0]
    return loss_1, loss_2


class LocalUpdate(object):
    def __init__(self, args, dataset=None, idxs=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)

    def train(self, net):
        net.train()
        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)

        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                if self.args.verbose and batch_idx % 10 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(images), len(self.ldr_train.dataset),
                              100. * batch_idx / len(self.ldr_train), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)


def init_local_update_objects_selfie(args, dataset_train, dict_users, noise_rates):
    local_update_objects = []
    for idx, noise_rate in zip(range(args.num_users), noise_rates):
        local_update_object = LocalUpdateSELFIE(
            args=args,
            dataset=dataset_train,
            idxs=dict_users[idx],
            noise_rate=noise_rate,
        )
        local_update_objects.append(local_update_object)

    return local_update_objects


class LocalUpdateSELFIE(object):
    def __init__(self, args, dataset=None, idxs=None, noise_rate=0):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss(reduction='none')
        self.ldr_train = DataLoader(
            DatasetSplit(dataset, idxs, real_idx_return=True),
            batch_size=self.args.local_bs,
            shuffle=True,
        )
        self.total_epochs = 0
        self.warmup = args.warmup_epochs
        self.corrector = SelfieCorrector(
            queue_size=args.queue_size,
            uncertainty_threshold=args.uncertainty_threshold,
            noise_rate=noise_rate,
            num_classes=args.num_classes,
        )

    def train(self, net):
        net.train()
        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)

        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, batch in enumerate(self.ldr_train):
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
                loss.backward()
                optimizer.step()

                if self.args.verbose and batch_idx % 10 == 0:
                    print(f"Update Epoch: {iter} [{batch_idx * len(images)}/{len(self.ldr_train.dataset)}"
                          f"({100. * batch_idx / len(self.ldr_train):.0f}%)]\tLoss: {loss.item():.6f}")

                batch_loss.append(loss.item())

            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            self.total_epochs += 1

        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)


class LocalUpdateJO(object):
    def __init__(self, args, dataset=None, idxs=None, results=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.dataset = dataset

        train_data_lst = []
        for idx in idxs:
            train_data_lst.append(dataset[idx])

        self.ldr_train = DataLoader(train_data_lst, batch_size=self.args.local_bs, shuffle=True)
        self.args = args
        self.results = results

    def train(self, net, g_epoch=0):
        net.train()
        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)

        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []

            for batch_idx, (images, labels, soft_targets, indexs) in enumerate(self.ldr_train):
                images, labels, soft_targets = images.to(self.args.device), labels.to(
                    self.args.device), soft_targets.to(self.args.device)
                outputs = net(images)

                probs, loss = self.mycriterion(outputs, soft_targets)

                if g_epoch >= self.args.begin:
                    self.dataset.soft_labels[
                        indexs.cpu().detach().numpy().tolist()] = probs.cpu().detach().numpy().tolist()

                net.zero_grad()
                loss.backward()
                optimizer.step()
                if self.args.verbose and batch_idx % 10 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(images), len(self.ldr_train.dataset),
                              100. * batch_idx / len(self.ldr_train), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        return net.state_dict(), sum(epoch_loss) / len(epoch_loss), self.results, self.dataset

    def mycriterion(self, outputs, soft_targets):
        # We introduce a prior probability distribution p, which is a distribution of classes among all training data.
        p = torch.ones(10).cuda() / 10

        probs = F.softmax(outputs, dim=1)
        avg_probs = torch.mean(probs, dim=0)

        L_c = -torch.mean(torch.sum(F.log_softmax(outputs, dim=1) * soft_targets, dim=1))
        L_p = -torch.sum(torch.log(avg_probs) * p)
        L_e = -torch.mean(torch.sum(F.log_softmax(outputs, dim=1) * probs, dim=1))

        loss = L_c + self.args.alpha * L_p + self.args.beta * L_e
        return probs, loss


class LocalUpdateCoteaching(object):
    def __init__(self, args, dataset=None, idxs=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss(reduce=False)
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)

    def train_coteaching(self, net1, net2, rate_schedule):
        net1.train()
        net2.train()
        # train and update
        optimizer1 = torch.optim.SGD(net1.parameters(), lr=self.args.lr, momentum=self.args.momentum,
                                     weight_decay=self.args.weight_decay)
        optimizer2 = torch.optim.SGD(net2.parameters(), lr=self.args.lr, momentum=self.args.momentum,
                                     weight_decay=self.args.weight_decay)

        epoch_loss1 = []
        epoch_loss2 = []
        for iter in range(self.args.local_ep):
            batch_loss1 = []
            batch_loss2 = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net1.zero_grad()
                net2.zero_grad()
                log_probs1 = net1(images)
                log_probs2 = net2(images)
                loss1, loss2 = loss_coteaching(log_probs1, log_probs2, labels, rate_schedule, self.loss_func)
                loss1.backward()
                loss2.backward()
                optimizer1.step()
                optimizer2.step()
                if self.args.verbose and batch_idx % 10 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss1: {:.6f}\tLoss2: {:.6f}'.format(
                        iter, batch_idx * len(images), len(self.ldr_train.dataset),
                              100. * batch_idx / len(self.ldr_train), loss1.item(), loss2.item()))
                batch_loss1.append(loss1.item())
                batch_loss2.append(loss2.item())
            epoch_loss1.append(sum(batch_loss1) / len(batch_loss1))
            epoch_loss2.append(sum(batch_loss2) / len(batch_loss2))
        return net1.state_dict(), sum(epoch_loss1) / len(epoch_loss1), net2.state_dict(), sum(epoch_loss2) / len(
            epoch_loss2)


class LocalUpdateCoteachingPlus(object):
    def __init__(self, args, dataset=None, idxs=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss(reduce=False)
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs, idx_return=True), batch_size=self.args.local_bs,
                                    shuffle=True)

    def train(self, net1, net2, rate_schedule, epoch):
        init_epoch = 10
        net1.train()
        net2.train()
        # train and update
        optimizer1 = torch.optim.SGD(net1.parameters(),
                                     lr=self.args.lr,
                                     momentum=self.args.momentum,
                                     weight_decay=self.args.weight_decay)
        optimizer2 = torch.optim.SGD(net2.parameters(),
                                     lr=self.args.lr,
                                     momentum=self.args.momentum,
                                     weight_decay=self.args.weight_decay)

        epoch_loss1 = []
        epoch_loss2 = []
        for iter in range(self.args.local_ep):
            batch_loss1 = []
            batch_loss2 = []
            for batch_idx, (images, labels, indexes) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net1.zero_grad()
                net2.zero_grad()
                log_probs1 = net1(images)
                log_probs2 = net2(images)
                if epoch < init_epoch:
                    loss1, loss2 = loss_coteaching(log_probs1,
                                                   log_probs2,
                                                   labels,
                                                   rate_schedule,
                                                   self.loss_func)
                else:
                    loss1, loss2 = loss_coteaching_plus(log_probs1,
                                                        log_probs2,
                                                        labels,
                                                        rate_schedule,
                                                        indexes,
                                                        epoch * batch_idx,
                                                        self.loss_func)
                loss1.backward()
                loss2.backward()
                optimizer1.step()
                optimizer2.step()
                if self.args.verbose and batch_idx % 10 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss1: {:.6f}\tLoss2: {:.6f}'.format(
                        iter, batch_idx * len(images), len(self.ldr_train.dataset),
                              100. * batch_idx / len(self.ldr_train), loss1.item(), loss2.item()))
                batch_loss1.append(loss1.item())
                batch_loss2.append(loss2.item())
            epoch_loss1.append(sum(batch_loss1) / len(batch_loss1))
            epoch_loss2.append(sum(batch_loss2) / len(batch_loss2))
        return net1.state_dict(), sum(epoch_loss1) / len(epoch_loss1), net2.state_dict(), sum(epoch_loss2) / len(
            epoch_loss2)


class LocalUpdateGFilter(object):
    def __init__(self, args, dataset=None, idxs=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss(reduce=False)
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)

    def train(self, net, forget_rate):
        net.train()
        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)

        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)
                loss = self.filter_loss(log_probs, labels, forget_rate)
                loss.backward()
                optimizer.step()
                if self.args.verbose and batch_idx % 10 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(images), len(self.ldr_train.dataset),
                              100. * batch_idx / len(self.ldr_train), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def filter_loss(self, y_pred, y_true, forget_rate):
        loss = self.loss_func(y_pred, y_true)
        ind_sorted = np.argsort(loss.data.cpu()).cuda()
        loss_sorted = loss[ind_sorted]

        remember_rate = 1 - forget_rate
        num_remember = int(remember_rate * len(loss_sorted))

        ind_update = ind_sorted[:num_remember]

        loss_update = self.loss_func(y_pred[ind_update], y_true[ind_update])

        return torch.sum(loss_update) / num_remember


class LocalUpdateLGFineTuning(object):
    def __init__(self, args, dataset=None, idxs=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.dataset = dataset
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs, idx_return=True), batch_size=self.args.local_bs,
                                    shuffle=True)

    def train(self, g_net, l_net, forget_rate):
        loss_func = nn.CrossEntropyLoss()
        loss_func_no_re = nn.CrossEntropyLoss(reduce=False)

        g_net.train()
        l_net.train()

        # train and update
        g_w = g_net.state_dict()
        l_w = l_net.state_dict()

        for key in g_w.keys():
            if 'fc3' not in key:
                l_w[key] = copy.deepcopy(g_w[key])

        l_net.load_state_dict(l_w)

        g_optimizer = torch.optim.SGD(g_net.parameters(), lr=self.args.lr, momentum=self.args.momentum,
                                      weight_decay=self.args.weight_decay)
        l_optimizer = torch.optim.SGD(l_net.parameters(), lr=self.args.lr, momentum=self.args.momentum,
                                      weight_decay=self.args.weight_decay)

        # fine-tuning
        for iter in range(self.args.ft_local_ep):
            for batch_idx, (images, labels, indexes) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                l_net.zero_grad()
                log_probs = l_net(images)
                loss = loss_func(log_probs, labels)
                loss.backward()
                l_optimizer.step()

        # Select clean data
        l_net.eval()
        loss_idx_lst = []
        for batch_idx, (images, labels, indexes) in enumerate(self.ldr_train):
            images, labels = images.to(self.args.device), labels.to(self.args.device)
            log_probs = l_net(images)
            loss = loss_func_no_re(log_probs, labels)

            for l, i in zip(loss, indexes):
                loss_idx_lst.append([l.item(), i.item()])

        loss_idx_lst.sort(key=lambda x: x[0])

        remember_rate = 1 - forget_rate
        num_remember = int(remember_rate * len(loss_idx_lst))

        filtered_train_data = []
        dataset = self.ldr_train.dataset

        for l, i in loss_idx_lst[:num_remember]:
            filtered_train_data.append((dataset[i][0], dataset[i][1]))

        trainloader = DataLoader(filtered_train_data, batch_size=self.args.local_bs, shuffle=True)

        epoch_loss = []

        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(trainloader):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                g_net.zero_grad()
                log_probs = g_net(images)
                loss = loss_func(log_probs, labels)
                loss.backward()
                g_optimizer.step()
                if self.args.verbose and batch_idx % 10 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(images), len(self.ldr_train.dataset),
                              100. * batch_idx / len(self.ldr_train), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
        return g_net.state_dict(), l_net.state_dict(), sum(epoch_loss) / len(epoch_loss)


class LocalUpdateFinetuning(object):
    def __init__(self, args, dataset=None, idxs=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.dataset = dataset
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs, idx_return=True), batch_size=self.args.local_bs,
                                    shuffle=True)

    def train(self, net, forget_rate):
        loss_func = nn.CrossEntropyLoss()
        loss_func_no_re = nn.CrossEntropyLoss(reduce=False)

        net.train()

        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)

        # train and update
        g_w = copy.deepcopy(net.state_dict())

        # fine-tuning
        for iter in range(self.args.ft_local_ep):
            for batch_idx, (images, labels, indexes) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)
                loss = loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()

        # Select clean data
        net.eval()

        loss_idx_lst = []
        for batch_idx, (images, labels, indexes) in enumerate(self.ldr_train):
            images, labels = images.to(self.args.device), labels.to(self.args.device)
            log_probs = net(images)
            loss = loss_func_no_re(log_probs, labels)

            for l, i in zip(loss, indexes):
                loss_idx_lst.append([l.item(), i.item()])

        loss_idx_lst.sort(key=lambda x: x[0])

        remember_rate = 1 - forget_rate
        num_remember = int(remember_rate * len(loss_idx_lst))

        filtered_train_data = []
        dataset = self.ldr_train.dataset

        for l, i in loss_idx_lst[:num_remember]:
            filtered_train_data.append((dataset[i][0], dataset[i][1]))

        trainloader = DataLoader(filtered_train_data, batch_size=int(self.args.local_bs * remember_rate), shuffle=True)

        net.load_state_dict(g_w)
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)

        epoch_loss = []

        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(trainloader):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)
                loss = loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                if self.args.verbose and batch_idx % 10 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(images), len(self.ldr_train.dataset),
                              100. * batch_idx / len(self.ldr_train), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)


class LocalUpdateBABU(object):
    def __init__(self, args, dataset=None, idxs=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)

    def train(self, net):
        net.train()
        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)

        epoch_loss = []

        body_params = [p for name, p in net.named_parameters() if 'fc3' not in name]
        head_params = [p for name, p in net.named_parameters() if 'fc3' in name]

        optimizer = torch.optim.SGD([{'params': body_params, 'lr': self.args.lr, 'momentum': self.args.momentum},
                                     {'params': head_params, 'lr': 0.0}])

        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                if self.args.verbose and batch_idx % 10 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(images), len(self.ldr_train.dataset),
                              100. * batch_idx / len(self.ldr_train), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)


class LocalUpdateGMix(object):
    def __init__(self, args, dataset=None, idxs=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs, idx_return=True), batch_size=self.args.local_bs,
                                    shuffle=True)

        self.num_iter = int(len(idxs) / args.local_bs)
        self.dataset = dataset

    def warmup(self, net):
        net.train()
        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)

        epoch_loss = []
        for ep in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels, idxs) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                if self.args.verbose and batch_idx % 10 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        ep, batch_idx * len(images), len(self.ldr_train.dataset),
                            100. * batch_idx / len(self.ldr_train), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def train(self, net, g_epoch, forget_rate=0.2):
        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)

        epoch_loss = []
        criterion = self.semi_loss

        loss_func_no_re = nn.CrossEntropyLoss(reduce=False)

        net.eval()

        loss_idx_lst = []
        for batch_idx, (images, labels, indexes) in enumerate(self.ldr_train):
            images, labels = images.to(self.args.device), labels.to(self.args.device)
            log_probs = net(images)
            loss = loss_func_no_re(log_probs, labels)

            for l, i in zip(loss, indexes):
                loss_idx_lst.append([l.item(), i.item()])

        loss_idx_lst.sort(key=lambda x: x[0])

        remember_rate = 1 - forget_rate
        num_remember = int(remember_rate * len(loss_idx_lst))

        # Divide dataset into label & unlabel 
        labeled_idxs = []
        unlabeled_idxs = []

        for idx, (l, i) in enumerate(loss_idx_lst):
            if idx < num_remember:
                labeled_idxs.append(i)
            else:
                unlabeled_idxs.append(i)

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

        labeled_trainloader = DataLoader(GMixLabeled(self.dataset, labeled_idxs),
                                         batch_size=self.args.local_bs,
                                         shuffle=True)

        unlabeled_trainloader = DataLoader(GMixUnlabeled(self.dataset, unlabeled_idxs),
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

                # MixMatch
                l = np.random.beta(self.args.mm_alpha, self.args.mm_alpha)

                l = max(l, 1 - l)

                all_inputs = torch.cat([inputs_x, inputs_u, inputs_u2], dim=0)
                all_targets = torch.cat([targets_x, targets_u, targets_u], dim=0)

                idx = torch.randperm(all_inputs.size(0))

                input_a, input_b = all_inputs, all_inputs[idx]

                target_a, target_b = all_targets, all_targets[idx]

                mixed_input = l * input_a + (1 - l) * input_b
                mixed_target = l * target_a + (1 - l) * target_b

                logits = net(mixed_input)
                logits_x = logits[:batch_size * 2]
                logits_u = logits[batch_size * 2:]

                Lx, Lu, lamb = criterion(logits_x, mixed_target[:batch_size * 2], logits_u,
                                         mixed_target[batch_size * 2:], g_epoch + batch_idx / self.num_iter,
                                         self.args.warm_up)
                # regularization
                prior = torch.ones(self.args.num_classes) / self.args.num_classes
                prior = prior.to(self.args.device)
                pred_mean = torch.softmax(logits, dim=1).mean(0)
                penalty = torch.sum(prior * torch.log(prior / pred_mean))

                loss = Lx + lamb * Lu + penalty

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

    def semi_loss(self, outputs_x, targets_x, outputs_u, targets_u, epoch, warm_up):
        probs_u = torch.softmax(outputs_u, dim=1)

        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        Lu = torch.mean((probs_u - targets_u) ** 2)

        return Lx, Lu, self._linear_rampup(epoch, warm_up)

    def _linear_rampup(self, current, warm_up, rampup_length=16):
        current = np.clip((current - warm_up) / rampup_length, 0.0, 1.0)
        return self.args.lambda_u * float(current)


class LocalUpdateDivideMix(object):
    def __init__(self, args, dataset=None, idxs=None):
        self.args = args
        self.dataset = dataset
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.idxs = idxs
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs, idx_return=True), batch_size=self.args.local_bs,
                                    shuffle=True)
        self.ldr_eval = DataLoader(DatasetSplit(dataset, idxs, real_idx_return=True), batch_size=self.args.local_bs)
        self.CE = nn.CrossEntropyLoss(reduction='none')
        self.CEloss = nn.CrossEntropyLoss()

    def warmup(self, net1, net2):
        net1.train()
        net2.train()

        # train and update
        optimizer1 = torch.optim.SGD(net1.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        optimizer2 = torch.optim.SGD(net2.parameters(), lr=self.args.lr, momentum=self.args.momentum)

        epoch_loss1 = []
        epoch_loss2 = []

        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels, idxs) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net1.zero_grad()
                log_probs = net1(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer1.step()
                batch_loss.append(loss.item())
            epoch_loss1.append(sum(batch_loss) / len(batch_loss))

        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels, idxs) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net2.zero_grad()
                log_probs = net2(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer2.step()
                batch_loss.append(loss.item())
            epoch_loss2.append(sum(batch_loss) / len(batch_loss))

        return net1.state_dict(), net2.state_dict()

    def train(self, g_epoch, warmup, net1, net2, all_loss):
        net1.train()
        net2.train()

        prob1, all_loss[0], prob_dict1, label_idx1, unlabel_idx1 = self.eval_train(net1, all_loss[0],
                                                                                   p_threshold=self.args.p_threshold)
        prob2, all_loss[1], prob_dict2, label_idx2, unlabel_idx2 = self.eval_train(net2, all_loss[1],
                                                                                   p_threshold=self.args.p_threshold)

        # train and update
        optimizer1 = torch.optim.SGD(net1.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        optimizer2 = torch.optim.SGD(net2.parameters(), lr=self.args.lr, momentum=self.args.momentum)

        epoch_loss1 = []
        epoch_loss2 = []

        # train net1
        labeled_trainloader = DataLoader(MixMatchLabeled(self.dataset, label_idx2, prob_dict2),
                                         batch_size=self.args.local_bs, shuffle=True)

        unlabeled_trainloader = DataLoader(MixMatchUnlabeled(self.dataset, unlabel_idx2),
                                           batch_size=self.args.local_bs, shuffle=True)
        net1.train()
        net2.eval()

        self.divide_mix(g_epoch, warmup, net1, net2, optimizer1, labeled_trainloader, unlabeled_trainloader)

        # train net2
        labeled_trainloader = DataLoader(MixMatchLabeled(self.dataset, label_idx1, prob_dict1),
                                         batch_size=self.args.local_bs, shuffle=True)

        unlabeled_trainloader = DataLoader(MixMatchUnlabeled(self.dataset, unlabel_idx1),
                                           batch_size=self.args.local_bs, shuffle=True)
        net1.eval()
        net2.train()

        self.divide_mix(g_epoch, warmup, net2, net1, optimizer2, labeled_trainloader, unlabeled_trainloader)

        return net1.state_dict(), net2.state_dict(), all_loss

    def divide_mix(self, epoch, warm_up, net, net2, optimizer, labeled_trainloader, unlabeled_trainloader):
        net.train()
        net2.eval()  # fix one network and train the other

        unlabeled_train_iter = iter(unlabeled_trainloader)
        labeled_train_iter = iter(labeled_trainloader)

        num_iter = int(len(self.idxs) / self.args.local_bs)

        criterion = SemiLoss()

        for batch_idx in range(num_iter):
            try:
                inputs_x, inputs_x2, labels_x, w_x = labeled_train_iter.next()
            except:
                labeled_train_iter = iter(labeled_trainloader)
                inputs_x, inputs_x2, labels_x, w_x = labeled_train_iter.next()

            try:
                inputs_u, inputs_u2 = unlabeled_train_iter.next()
            except:
                unlabeled_train_iter = iter(unlabeled_trainloader)
                inputs_u, inputs_u2 = unlabeled_train_iter.next()

            batch_size = inputs_x.size(0)

            # Transform label to one-hot
            labels_x = torch.zeros(batch_size, self.args.num_classes).scatter_(1, labels_x.view(-1, 1), 1)
            w_x = w_x.view(-1, 1).type(torch.FloatTensor)

            inputs_x, inputs_x2, labels_x, w_x = inputs_x.cuda(), inputs_x2.cuda(), labels_x.cuda(), w_x.cuda()
            inputs_u, inputs_u2 = inputs_u.cuda(), inputs_u2.cuda()

            with torch.no_grad():
                # label co-guessing of unlabeled samples
                outputs_u11 = net(inputs_u)
                outputs_u12 = net(inputs_u2)
                outputs_u21 = net2(inputs_u)
                outputs_u22 = net2(inputs_u2)

                pu = (torch.softmax(outputs_u11, dim=1) + torch.softmax(outputs_u12, dim=1) + torch.softmax(outputs_u21,
                                                                                                            dim=1) + torch.softmax(
                    outputs_u22, dim=1)) / 4
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
            l = np.random.beta(self.args.mm_alpha, self.args.mm_alpha)
            l = max(l, 1 - l)

            all_inputs = torch.cat([inputs_x, inputs_x2, inputs_u, inputs_u2], dim=0)
            all_targets = torch.cat([targets_x, targets_x, targets_u, targets_u], dim=0)

            idx = torch.randperm(all_inputs.size(0))

            input_a, input_b = all_inputs, all_inputs[idx]
            target_a, target_b = all_targets, all_targets[idx]

            mixed_input = l * input_a + (1 - l) * input_b
            mixed_target = l * target_a + (1 - l) * target_b

            logits = net(mixed_input)
            logits_x = logits[:batch_size * 2]
            logits_u = logits[batch_size * 2:]

            Lx, Lu, lamb = criterion(self.args, logits_x, mixed_target[:batch_size * 2], logits_u,
                                     mixed_target[batch_size * 2:], epoch + batch_idx / num_iter, warm_up)

            # regularization
            prior = torch.ones(self.args.num_classes) / self.args.num_classes
            prior = prior.cuda()
            pred_mean = torch.softmax(logits, dim=1).mean(0)
            penalty = torch.sum(prior * torch.log(prior / pred_mean))

            loss = Lx + lamb * Lu + penalty
            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def eval_train(self, model, all_loss, p_threshold):
        model.eval()
        losses_lst = []
        idx_lst = []

        with torch.no_grad():
            for batch_idx, (inputs, targets, items, idxs) in enumerate(self.ldr_eval):
                inputs, targets = inputs.cuda(), targets.cuda()
                outputs = model(inputs)
                losses_lst += [l.item() for l in self.CE(outputs, targets)]

                idx_lst += [i.item() for i in idxs]

        losses = torch.torch.FloatTensor(losses_lst)
        losses = (losses - losses.min()) / (losses.max() - losses.min())
        all_loss.append(losses)

        input_loss = losses.reshape(-1, 1)

        # fit a two-component GMM to the loss
        gmm = GaussianMixture(n_components=2, max_iter=100, tol=1e-2, reg_covar=5e-4)
        gmm.fit(input_loss)
        prob = gmm.predict_proba(input_loss)
        prob = prob[:, gmm.means_.argmin()]

        prob_dict = {}

        for i, p in zip(idx_lst, prob):
            prob_dict[i] = p

        pred = (prob > p_threshold)
        label_idx = pred.nonzero()[0]
        unlabel_idx = (1 - pred).nonzero()[0]

        label_idx = [idx_lst[i] for i in label_idx]
        unlabel_idx = [idx_lst[i] for i in unlabel_idx]

        return prob, all_loss, prob_dict, label_idx, unlabel_idx


class LocalUpdateHS(object):
    def __init__(self, args, dataset=None, idxs=None, results=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.loss_func_no_red = nn.CrossEntropyLoss(reduction="none")
        self.dataset = dataset
        self.args = args
        self.results = results
        self.idxs = idxs

    def warmup(self, net, s_cnt):
        net.train()

        ldr_train = DataLoader(DatasetSplitHS(self.dataset, self.idxs), batch_size=self.args.local_bs, shuffle=True)

        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)

        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels, soft_labels, predictions, indexes) in enumerate(ldr_train):
                images, labels, soft_labels = images.to(self.args.device), labels.to(self.args.device), soft_labels.to(
                    self.args.device)
                net.zero_grad()
                log_probs = net(images)

                loss = self.loss_func(log_probs, labels)

                if iter == 0:
                    for prob, index in zip(log_probs, indexes):
                        self.dataset.prediction[index][s_cnt % self.args.K] = F.softmax(prob.cpu()).detach().numpy()

                loss.backward()
                optimizer.step()

                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        return net.state_dict(), sum(epoch_loss) / len(epoch_loss), self.dataset

    def train(self, net, s_cnt, forget_rate=0.2):
        # Select clean data
        net.eval()

        ldr_train = DataLoader(DatasetSplitHS(self.dataset, self.idxs), batch_size=self.args.local_bs, shuffle=True)

        loss_idx_lst = []
        for batch_idx, (images, labels, soft_labels, pred_lst, indexes) in enumerate(ldr_train):
            images, labels = images.to(self.args.device), labels.to(self.args.device)
            log_probs = net(images)
            loss = self.loss_func_no_red(log_probs, labels)

            for l, i, prob in zip(loss, indexes, log_probs):
                self.dataset.prediction[i][s_cnt % self.args.K] = prob.cpu().detach().numpy()
                loss_idx_lst.append([l.item(), i.item()])

        loss_idx_lst.sort(key=lambda x: x[0])

        remember_rate = 1 - forget_rate
        num_remember = int(remember_rate * len(loss_idx_lst))

        # pseudo label (Hard Label)
        for l, i in loss_idx_lst[num_remember:]:
            self.dataset.train_labels[i] = int(np.argmax(np.mean(self.dataset.prediction[i], axis=0)))

        ldr_train = DataLoader(DatasetSplitHS(self.dataset, self.idxs), batch_size=self.args.local_bs, shuffle=True)
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)

        net.train()

        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels, soft_labels, predictions, indexes) in enumerate(ldr_train):
                images, labels, soft_labels = images.to(self.args.device), labels.to(self.args.device), soft_labels.to(
                    self.args.device)
                net.zero_grad()
                log_probs = net(images)

                loss = self.loss_func(log_probs, labels)

                if iter == 0:
                    for prob, index in zip(log_probs, indexes):
                        self.dataset.prediction[index][s_cnt % self.args.K] = prob.cpu().detach().numpy()

                loss.backward()
                optimizer.step()

                if self.args.verbose and batch_idx % 10 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(images), len(self.ldr_train.dataset),
                              100. * batch_idx / len(self.ldr_train), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss), self.dataset

    def criterion_softlabel(self, outputs, soft_targets):
        return self.loss_func(F.softmax(outputs), soft_targets)
