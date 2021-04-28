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


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs, idx_return=False):
        self.dataset = dataset
        self.idxs = list(idxs)
        self.idx_return = idx_return

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        item = int(item)
        image, label = self.dataset[self.idxs[item]]
        
        if self.idx_return:
            return image, label, item
        else:
            return image, label

        
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

    ind_1_update=ind_1_sorted[:num_remember]
    ind_2_update=ind_2_sorted[:num_remember]
    # exchange
    loss_1_update = loss_func(y_pred1[ind_2_update], y_true[ind_2_update])
    loss_2_update = loss_func(y_pred2[ind_1_update], y_true[ind_1_update])

    return torch.sum(loss_1_update)/num_remember, torch.sum(loss_2_update)/num_remember
    
    
def loss_coteaching_plus(logits, logits2, labels, forget_rate, ind, step, loss_func=None):
    outputs = F.softmax(logits, dim=1)
    outputs2 = F.softmax(logits2, dim=1)

    _, pred1 = torch.max(logits.data, 1)
    _, pred2 = torch.max(logits2.data, 1)

    pred1, pred2 = pred1.cpu().numpy(), pred2.cpu().numpy()

    logical_disagree_id=np.zeros(labels.size(), dtype=bool)
    disagree_id = []
    for idx, p1 in enumerate(pred1): 
        if p1 != pred2[idx]:
            disagree_id.append(idx) 
            logical_disagree_id[idx] = True
    
    temp_disagree = ind*logical_disagree_id.astype(np.int64)
    ind_disagree = np.asarray([i for i in temp_disagree if i != 0]).transpose()
    try:
        assert ind_disagree.shape[0]==len(disagree_id)
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

        loss_1 = torch.sum(update_step*cross_entropy_1)/labels.size()[0]
        loss_2 = torch.sum(update_step*cross_entropy_2)/labels.size()[0]
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
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
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
                images, labels, soft_targets = images.to(self.args.device), labels.to(self.args.device), soft_targets.to(self.args.device)
                outputs = net(images)
                
                probs, loss = self.mycriterion(outputs, soft_targets)
                
                if g_epoch >= self.args.begin:
                    self.dataset.soft_labels[indexs.cpu().detach().numpy().tolist()] = probs.cpu().detach().numpy().tolist()
                
                net.zero_grad()
                loss.backward()
                optimizer.step()
                if self.args.verbose and batch_idx % 10 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(images), len(self.ldr_train.dataset),
                               100. * batch_idx / len(self.ldr_train), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
            
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
        optimizer1 = torch.optim.SGD(net1.parameters(), lr=self.args.lr, momentum=self.args.momentum, weight_decay=self.args.weight_decay)
        optimizer2 = torch.optim.SGD(net2.parameters(), lr=self.args.lr, momentum=self.args.momentum, weight_decay=self.args.weight_decay)

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
            epoch_loss1.append(sum(batch_loss1)/len(batch_loss1))
            epoch_loss2.append(sum(batch_loss2)/len(batch_loss2))
        return net1.state_dict(), sum(epoch_loss1) / len(epoch_loss1), net2.state_dict(), sum(epoch_loss2) / len(epoch_loss2)

    
class LocalUpdateCoteachingPlus(object):
    def __init__(self, args, dataset=None, idxs=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss(reduce=False)
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs, idx_return=True), batch_size=self.args.local_bs, shuffle=True)
        
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
                                                          epoch*batch_idx, 
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
            epoch_loss1.append(sum(batch_loss1)/len(batch_loss1))
            epoch_loss2.append(sum(batch_loss2)/len(batch_loss2))
        return net1.state_dict(), sum(epoch_loss1) / len(epoch_loss1), net2.state_dict(), sum(epoch_loss2) / len(epoch_loss2)
    
    
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
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)
    
    def filter_loss(self, y_pred, y_true, forget_rate):
        loss = self.loss_func(y_pred, y_true)
        ind_sorted = np.argsort(loss.data.cpu()).cuda()
        loss_sorted = loss[ind_sorted]

        remember_rate = 1 - forget_rate
        num_remember = int(remember_rate * len(loss_sorted))

        ind_update=ind_sorted[:num_remember]
        
        loss_update = self.loss_func(y_pred[ind_update], y_true[ind_update])
        
        return torch.sum(loss_update) / num_remember

    
class LocalUpdateLGFineTuning(object):
    def __init__(self, args, dataset=None, idxs=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.dataset = dataset
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs, idx_return=True), batch_size=self.args.local_bs, shuffle=True)
        
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
        
        g_optimizer = torch.optim.SGD(g_net.parameters(), lr=self.args.lr, momentum=self.args.momentum, weight_decay=self.args.weight_decay)
        l_optimizer = torch.optim.SGD(l_net.parameters(), lr=self.args.lr, momentum=self.args.momentum, weight_decay=self.args.weight_decay)

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
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return g_net.state_dict(), l_net.state_dict(), sum(epoch_loss) / len(epoch_loss)

    
class LocalUpdateFinetuning(object):
    def __init__(self, args, dataset=None, idxs=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.dataset = dataset
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs, idx_return=True), batch_size=self.args.local_bs, shuffle=True)
        
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
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
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
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)
    
    
class LocalUpdateGMix(object):
    def __init__(self, args, dataset=None, idxs=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs, idx_return=True), batch_size=self.args.local_bs, shuffle=True)
        
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
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)
    
    def train(self, net, g_epoch):
        net.eval()
        
        # Divide dataset into label & unlabel 
        labeled_idxs = []
        unlabeled_idxs = []
        
        for batch_idx, (images, labels, idxs) in enumerate(self.ldr_train):
            images, labels = images.to(self.args.device), labels.to(self.args.device)
            log_probs = net(images)
            
            y_pred = log_probs.data.max(1, keepdim=True)[1]
            
            for match, idx in zip(y_pred.eq(labels.data.view_as(y_pred)), idxs):
                if match[0].item():
                    labeled_idxs.append(idx.item())
                else:
                    unlabeled_idxs.append(idx.item())
        
        if len(labeled_idxs) % 2 != 0:
            unlabeled_idxs.append(labeled_idxs.pop(-1))
                    
        
        labeled_trainloader = DataLoader(GMixLabeled(self.dataset, labeled_idxs), batch_size=self.args.local_bs, shuffle=True)
        unlabeled_trainloader = DataLoader(GMixUnlabeled(self.dataset, unlabeled_idxs), batch_size=self.args.local_bs, shuffle=True)
        
        labeled_train_iter = iter(labeled_trainloader)
        unlabeled_train_iter = iter(unlabeled_trainloader)
        
        net.train()
        
        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        
        epoch_loss = []
        criterion = self.semi_loss
        
        for ep in range(self.args.local_ep):
            batch_loss = []
            
            for batch_idx in range(self.num_iter):
                try:
                    inputs_x, inputs_x2, targets_x = labeled_train_iter.next()
                except:
                    labeled_train_iter = iter(labeled_trainloader)
                    inputs_x, inputs_x2, targets_x = labeled_train_iter.next()

                try:
                    inputs_u, inputs_u2 = unlabeled_train_iter.next()
                except:
                    unlabeled_train_iter = iter(unlabeled_trainloader)
                    inputs_u, inputs_u2 = unlabeled_train_iter.next()
                
                batch_size = inputs_x.size(0)
                
                # Transform label to one-hot
                targets_x = torch.zeros(batch_size, 10).scatter_(1, targets_x.view(-1,1).long(), 1)
                
                inputs_x, targets_x = inputs_x.to(self.args.device), targets_x.to(self.args.device)
                inputs_u, inputs_u2 = inputs_u.to(self.args.device), inputs_u2.to(self.args.device)
                
                with torch.no_grad():
                    outputs_u = net(inputs_u)
                    outputs_u2 = net(inputs_u2)
                    p = (torch.softmax(outputs_u, dim=1) + torch.softmax(outputs_u2, dim=1)) / 2
                    pt = p**(1/self.args.T)
                    targets_u = pt / pt.sum(dim=1, keepdim=True)
                    targets_u = targets_u.detach()
                
                # MixUp
                all_inputs = torch.cat([inputs_x, inputs_u, inputs_u2], dim=0)
                all_targets = torch.cat([targets_x, targets_u, targets_u], dim=0)
                
                l = np.random.beta(self.args.mm_alpha, self.args.mm_alpha)

                l = max(l, 1-l)

                idx = torch.randperm(all_inputs.size(0))
                
                input_a, input_b = all_inputs, all_inputs[idx]
                
                target_a, target_b = all_targets, all_targets[idx]
                
                mixed_input = l * input_a + (1 - l) * input_b
                mixed_target = l * target_a + (1 - l) * target_b
                
                # interleave labeled and unlabed samples between batches to get correct batchnorm calculation 
                mixed_input = list(torch.split(mixed_input, batch_size))
                mixed_input = self.interleave(mixed_input, batch_size)

                logits = [net(mixed_input[0])]
                for input in mixed_input[1:]:
                    logits.append(net(input))

                # put interleaved samples back
                logits = self.interleave(logits, batch_size)
                logits_x = logits[0]
                logits_u = torch.cat(logits[1:], dim=0)
                
                print(logits_x.shape, mixed_target.shape, mixed_target[:batch_size].shape)

                Lx, Lu, w = criterion(logits_x, mixed_target[:batch_size], logits_u, mixed_target[batch_size:], g_epoch + batch_idx / self.num_iter)

                loss = Lx + w * Lu
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                if self.args.verbose and batch_idx % 10 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        ep, batch_idx * len(images), len(self.ldr_train.dataset),
                               100. * batch_idx / len(self.ldr_train), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)
    
    def semi_loss(self, outputs_x, targets_x, outputs_u, targets_u, epoch):
        probs_u = torch.softmax(outputs_u, dim=1)

        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        Lu = torch.mean((probs_u - targets_u)**2)

        return Lx, Lu, self.args.lambda_u * self._linear_rampup(epoch, rampup_length=self.args.epochs)
        
    def interleave(self, xy, batch):
        nu = len(xy) - 1
        offsets = self._interleave_offsets(batch, nu)
        xy = [[v[offsets[p]:offsets[p + 1]] for p in range(nu + 1)] for v in xy]
        for i in range(1, nu + 1):
            xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
        return [torch.cat(v, dim=0) for v in xy]
    
    def _interleave_offsets(self, batch, nu):
        groups = [batch // (nu + 1)] * (nu + 1)
        for x in range(batch - sum(groups)):
            groups[-x - 1] += 1
        offsets = [0]
        for g in groups:
            offsets.append(offsets[-1] + g)
        assert offsets[-1] == batch
        return offsets

    def _linear_rampup(self, current, rampup_length):
        if rampup_length == 0:
            return 1.0
        else:
            current = np.clip(current / rampup_length, 0.0, 1.0)
            return float(current)


