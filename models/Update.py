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
    

class RFLloss:
    def __call__(self, logit, labels, idx, feature, f_k, pseudo_labels, mask, small_loss_idxs, lambda_cen, lambda_e):
        mse = torch.nn.MSELoss()
        ce = torch.nn.CrossEntropyLoss()
        
        L_c = torch.sum(mask[small_loss_idxs] * ce(logit[small_loss_idxs], labels[small_loss_idxs]) + (1-mask[small_loss_idxs]) * ce(logit[small_loss_idxs], pseudo_labels[idx[small_loss_idxs]]))
        L_cen = torch.sum(mask[small_loss_idxs] * mse(feature[small_loss_idxs], f_k[labels[[small_loss_idxs]]]))
        L_e = -torch.sum(F.softmax(logit[small_loss_idxs]) * F.log_softmax(logit[small_loss_idxs]))

        return L_c + lambda_cen * L_cen + lambda_e * L_e


def get_local_update_objects(args, dataset_train, dict_users=None, noise_rates=None, net_glob=None, f_G=None):
    local_update_objects = []
    for idx, noise_rate in zip(range(args.num_users), noise_rates):
        local_update_args = dict(
            args=args,
            dataset=dataset_train,
            idxs=dict_users[idx],
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

        elif args.method == 'RFL':
            local_update_object = LocalUpdateRFL(f_G=f_G, **local_update_args)
            
        local_update_objects.append(local_update_object)

    return local_update_objects


class BaseLocalUpdate:
    def __init__(self, args, dataset=None, idxs=None, idx_return=False, real_idx_return=False, is_babu=False):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()

        self.dataset = dataset
        self.idxs = idxs

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

    def train(self, net, net2=None):
        if net2 is None:
            return self.train_single_model(net)
        else:
            return self.train_multiple_models(net, net2)

    def train_single_model(self, net):
        net.train()

        if self.is_babu:
            body_params = [p for name, p in net.named_parameters() if 'fc3' not in name]
            head_params = [p for name, p in net.named_parameters() if 'fc3' in name]

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

        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def train_multiple_models(self, net1, net2):
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

        return net1.state_dict(), sum(epoch_loss1) / len(epoch_loss1), \
               net2.state_dict(), sum(epoch_loss2) / len(epoch_loss2)

    def forward_pass(self, batch, net, net2=None):
        if self.idx_return:
            images, labels, _ = batch

        elif self.real_idx_return:
            images, labels, _, _ = batch

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
        pass

    
class LocalUpdateRFL(BaseLocalUpdate):
    def __init__(self, args, dataset=None, idxs=None, f_G=None, idx_return=True):
        super().__init__(
            args=args,
            dataset=dataset,
            idxs=idxs,
        )
        self.f_G = f_G
        self.pseudo_labels = torch.zeros(len(self.dataset), dtype=torch.long, device=self.args.device)
        self.sim = torch.nn.CosineSimilarity(dim=1, eps=1e-6) 
        self.loss = RFLloss()
        self.loss_func = torch.nn.CrossEntropyLoss(reduce=False)
        self.ldr_train = DataLoader(
            DatasetSplit(dataset, idxs, idx_return=idx_return),
            batch_size=self.args.local_bs,
            shuffle=True,
        )
        self.ldr_train_tmp = DataLoader(DatasetSplit(dataset, idxs, idx_return=idx_return), batch_size=1, shuffle=True)
        
             
    def get_small_loss_samples(self, y_pred, y_true, forget_rate):
        loss = self.loss_func(y_pred, y_true)
        ind_sorted = np.argsort(loss.data.cpu()).cuda()
        loss_sorted = loss[ind_sorted]

        remember_rate = 1 - forget_rate
        num_remember = int(remember_rate * len(loss_sorted))

        ind_update=ind_sorted[:num_remember]

        return ind_update
        
    def train(self, net):
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum, weight_decay=self.args.weight_decay)
        epoch_loss = []
        f_k = torch.zeros(10, 128, device=self.args.device)
        # initialization of global centroids
        net.eval()
        # obtain naive average feature
        n_labels = torch.zeros(10, 1, device=self.args.device)
        if self.args.g_epoch == 0:
            with torch.no_grad():
                for batch_idx, (images, labels, idxs) in enumerate(self.ldr_train_tmp):
                    net.zero_grad()
                    images, labels = images.to(self.args.device), labels.to(self.args.device)
                    logit, feature = net(images)
                    f_k[labels] = f_k[labels] + feature
                    n_labels[labels] = n_labels[labels] + 1
                
            f_k = torch.div(f_k, n_labels)# .to(self.args.device)
        else:
            f_k = f_G
        
        # obtain global-guided pseudo labels y_hat by y_hat_k = C_G(F_G(x_k))
        with torch.no_grad():
            for batch_idx, (images, labels, idxs) in enumerate(self.ldr_train_tmp):
                net.zero_grad()
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                logit, feature = net(images)
                self.pseudo_labels[idxs] = torch.argmax(logit)

        net.train()
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, batch in enumerate(self.ldr_train):
                net.zero_grad()        
                images, labels, idx = batch
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                logit, feature = net(images)
                f_k = f_k.to(self.args.device)
                # batch 안에서의 index 순서
                small_loss_idxs = self.get_small_loss_samples(logit, labels, self.args.forget_rate)

                y_k_tilde = torch.zeros(self.args.local_bs, device=self.args.device)
                for i in small_loss_idxs:
                    y_k_tilde[i] = torch.argmax(self.sim(f_k, torch.reshape(feature[i], (1, 128)).clone()))

                mask = torch.zeros(self.args.local_bs, device=self.args.device)
                for i in small_loss_idxs:
                    if y_k_tilde[i] == labels[i]:
                        mask[i] = 1
                
                if self.args.g_epoch < self.args.T_pl:
                    for i in small_loss_idxs:    
                        self.pseudo_labels[idx[i]] = labels[i]

                loss = self.loss(logit, labels, idx, feature, f_k, self.pseudo_labels, mask, small_loss_idxs, self.args.lambda_cen, self.args.lambda_e)
               
                #labels, feature = labels.to(self.args.device), feature.to(self.args.device)
                # weight update by minimizing loss: L_total = L_c + lambda_cen * L_cen + lambda_e * L_e
                loss.backward(retain_graph=True)
                optimizer.step()
                
                # obtain loss based average features f_k,j_hat from small loss dataset
                f_kj_hat = torch.zeros(10, 128, device=self.args.device)
                n = torch.ones(10, 1, device=self.args.device)
                for i in small_loss_idxs:
                    f_kj_hat[labels[i]] = f_kj_hat[labels[i]] + feature[i]
                    n[labels[i]] = n[labels[i]] + 1
                f_kj_hat = torch.div(f_kj_hat, n)
                
                # update local centroid f_k
                
                one = torch.ones(10, 1, device=self.args.device)
                f_k = (one - self.sim(f_k, f_kj_hat).reshape(10, 1) ** 2) * f_k + (self.sim(f_k, f_kj_hat).reshape(10, 1) ** 2) * f_kj_hat

                if self.args.verbose and batch_idx % 10 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(images), len(self.ldr_train.dataset),
                               100. * batch_idx / len(self.ldr_train), loss.item()))
                batch_loss.append(loss.item())

            epoch_loss.append(sum(batch_loss)/len(batch_loss))
            
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss), f_k         

    
    
class LocalUpdateFedProx(BaseLocalUpdate):
    def __init__(self, args, dataset=None, idxs=None):
        super().__init__(
            args=args,
            dataset=dataset,
            idxs=idxs,
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
    def __init__(self, args, dataset=None, idxs=None, noise_rate=0):
        super().__init__(
            args=args,
            dataset=dataset,
            idxs=idxs,
            real_idx_return=True
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
        return loss


class LocalUpdateJointOptim(BaseLocalUpdate):
    def __init__(self, args, dataset=None, idxs=None):
        super().__init__(
            args=args,
            dataset=dataset,
            idxs=idxs,
            real_idx_return=True
        )
        self.stop_update_epoch = args.stop_update_epoch
        self.corrector = JointOptimCorrector(
            queue_size=args.K,
            num_classes=args.num_classes,
            data_size=len(idxs),
        )

    def forward_pass(self, batch, net, net2=None):
        images, labels, _, ids = batch
        ids = ids.numpy()

        _, soft_labels = self.corrector.get_labels(ids, labels)
        soft_labels = soft_labels.to(self.args.device)
        images = images.to(self.args.device)

        logits = net(images)
        probs = F.softmax(logits, dim=1)

        is_loss_cross_entropy = self.total_epochs >= self.stop_update_epoch
        loss = self.joint_optim_loss(logits, probs, soft_labels, is_cross_entropy=is_loss_cross_entropy)

        self.corrector.update_probability_history(ids, probs.cpu().detach())
        return loss

    def on_epoch_end(self):
        if self.args.begin_update_epoch <= self.total_epochs < self.args.stop_update_epoch:
            self.corrector.update_labels()

    def joint_optim_loss(self, logits, probs, soft_targets, is_cross_entropy=False):
        if is_cross_entropy:
            loss = -torch.mean(torch.sum(F.log_softmax(logits, dim=1) * soft_targets, dim=1))

        else:
            # We introduce a prior probability distribution p,
            # which is a distribution of classes among all training data.
            p = torch.ones(self.args.num_classes).cuda() / self.args.num_classes

            avg_probs = torch.mean(probs, dim=0)

            L_c = -torch.mean(torch.sum(F.log_softmax(logits, dim=1) * soft_targets, dim=1))
            L_p = -torch.sum(torch.log(avg_probs) * p)
            L_e = -torch.mean(torch.sum(F.log_softmax(logits, dim=1) * probs, dim=1))

            loss = L_c + self.args.alpha * L_p + self.args.beta * L_e

        return loss


class LocalUpdateGFilter(BaseLocalUpdate):
    def __init__(self, args, dataset=None, idxs=None):
        super().__init__(
            args=args,
            dataset=dataset,
            idxs=idxs,
        )
        self.loss_func = nn.CrossEntropyLoss(reduce=False)

    def forward_pass(self, batch, net, net2=None):
        images, labels = batch
        images = images.to(self.args.device)
        labels = labels.to(self.args.device)

        log_probs = net(images)
        loss = self.filter_loss(log_probs, labels, self.args.forget_rate)

        return loss

    def filter_loss(self, y_pred, y_true, forget_rate):
        loss = self.loss_func(y_pred, y_true)
        ind_sorted = np.argsort(loss.data.cpu()).cuda()
        loss_sorted = loss[ind_sorted]

        remember_rate = 1 - forget_rate
        num_remember = int(remember_rate * len(loss_sorted))

        ind_update = ind_sorted[:num_remember]

        loss_update = self.loss_func(y_pred[ind_update], y_true[ind_update])

        return torch.sum(loss_update) / num_remember


class LocalUpdateCoteaching(BaseLocalUpdate):
    def __init__(self, args, dataset=None, idxs=None, is_coteaching_plus=False):
        super().__init__(
            args=args,
            dataset=dataset,
            idxs=idxs,
            idx_return=is_coteaching_plus
        )
        self.loss_func = nn.CrossEntropyLoss(reduce=False)
        self.is_coteaching_plus = is_coteaching_plus

        self.init_epoch = 10  # only used for coteaching+

    def forward_pass(self, batch, net, net2=None):
        if self.is_coteaching_plus:
            images, labels, indices = batch
        else:
            images, labels = batch
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
            loss1, loss2 = self.loss_coteaching_plus(indices=indices, step=self.epoch * self.batch_idx, **loss_args)
        else:
            loss1, loss2 = self.loss_coteaching(**loss_args)

        return loss1, loss2

    def loss_coteaching(self, y_pred1, y_pred2, y_true, forget_rate):
        loss_1 = self.loss_func(y_pred1, y_true)
        ind_1_sorted = np.argsort(loss_1.data.cpu()).cuda()
        loss_1_sorted = loss_1[ind_1_sorted]

        loss_2 = self.loss_func(y_pred2, y_true)
        ind_2_sorted = np.argsort(loss_2.data.cpu()).cuda()
        loss_2_sorted = loss_2[ind_2_sorted]

        remember_rate = 1 - forget_rate
        num_remember = int(remember_rate * len(loss_1_sorted))

        ind_1_update = ind_1_sorted[:num_remember]
        ind_2_update = ind_2_sorted[:num_remember]
        # exchange
        loss_1_update = self.loss_func(y_pred1[ind_2_update], y_true[ind_2_update])
        loss_2_update = self.loss_func(y_pred2[ind_1_update], y_true[ind_1_update])

        return torch.sum(loss_1_update) / num_remember, torch.sum(loss_2_update) / num_remember

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
            loss_1, loss_2, = self.loss_coteaching(update_outputs, update_outputs2, update_labels, forget_rate)
        else:
            update_step = np.logical_or(logical_disagree_id, step < 5000).astype(np.float32)
            update_step = Variable(torch.from_numpy(update_step)).cuda()

            cross_entropy_1 = F.cross_entropy(outputs, y_true)
            cross_entropy_2 = F.cross_entropy(outputs2, y_true)

            loss_1 = torch.sum(update_step * cross_entropy_1) / y_true.size()[0]
            loss_2 = torch.sum(update_step * cross_entropy_2) / y_true.size()[0]
        return loss_1, loss_2


class LocalUpdateLGteaching(LocalUpdateCoteaching):
    def __init__(self, args, dataset=None, idxs=None, l_net=None):
        super().__init__(
            args=args,
            dataset=dataset,
            idxs=idxs,
        )
        self.l_net = copy.deepcopy(l_net)

    def train(self, net, net2=None):
        w_g, loss1, w_l, loss2 = self.train_multiple_models(net, self.l_net)
        return w_g, loss1


class LocalUpdateLGFineTuning(BaseLocalUpdate):
    def __init__(self, args, dataset=None, idxs=None, l_net=None):
        super().__init__(
            args=args,
            dataset=dataset,
            idxs=idxs,
        )
        self.loss_func_no_re = nn.CrossEntropyLoss(reduce=False)
        self.finetuning_train = DataLoader(
            DatasetSplit(dataset, idxs, idx_return=True),
            batch_size=self.args.local_bs,
            shuffle=True,
        )
        self.l_net = copy.deepcopy(l_net)

    def train(self, net, net2=None):
        g_net = net
        l_net = self.l_net

        g_net.train()
        l_net.train()

        # train and update
        g_w = g_net.state_dict()
        l_w = l_net.state_dict()

        for key in g_w.keys():
            if 'fc3' not in key:
                l_w[key] = copy.deepcopy(g_w[key])

        l_net.load_state_dict(l_w)
        l_optimizer = torch.optim.SGD(l_net.parameters(), lr=self.args.lr, momentum=self.args.momentum,
                                      weight_decay=self.args.weight_decay)

        # fine-tuning
        for iter in range(self.args.ft_local_ep):
            for batch_idx, (images, labels, indexes) in enumerate(self.finetuning_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                l_net.zero_grad()
                log_probs = l_net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                l_optimizer.step()

        # Select clean data
        l_net.eval()
        loss_idx_lst = []
        for batch_idx, (images, labels, indexes) in enumerate(self.finetuning_train):
            images, labels = images.to(self.args.device), labels.to(self.args.device)
            log_probs = l_net(images)
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
        return self.train_single_model(g_net)


class LocalUpdateFinetuning(BaseLocalUpdate):
    def __init__(self, args, dataset=None, idxs=None):
        super().__init__(
            args=args,
            dataset=dataset,
            idxs=idxs,
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
    def __init__(self, args, dataset=None, idxs=None):
        super().__init__(
            args=args,
            dataset=dataset,
            idxs=idxs,
            idx_return=True
        )
        self.num_iter = int(len(idxs) / args.local_bs)
        self.semiloss = SemiLoss()

    def train(self, net, net2=None):
        if self.args.g_epoch <= self.args.warm_up:
            return self.train_single_model(net)
        else:
            return self.train_2_phase(net)

    def train_2_phase(self, net):
        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)

        epoch_loss = []
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
                    warm_up=self.args.warm_up,
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
    def __init__(self, args, dataset=None, idxs=None):
        super().__init__(
            args=args,
            dataset=dataset,
            idxs=idxs,
            idx_return=True,
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

    def train(self, net, net2=None):
        if self.args.g_epoch <= self.args.warm_up:
            return self.train_multiple_models(net, net2)
        else:
            return self.train_2_phase(net, net2)

    def train_2_phase(self, net, net2):
        prob_dict1, label_idx1, unlabel_idx1 = self.update_probabilties_split_data_indices(net, self.loss_history1)
        prob_dict2, label_idx2, unlabel_idx2 = self.update_probabilties_split_data_indices(net2, self.loss_history2)

        # train net1
        loss1 = self.divide_mix(
            net=net,
            net2=net2,
            label_idx=label_idx2,
            prob_dict=prob_dict2,
            unlabel_idx=unlabel_idx2,
            warm_up=self.args.warm_up,
            epoch=self.args.g_epoch,
        )

        # train net2
        loss2 = self.divide_mix(
            net=net2,
            net2=net,
            label_idx=label_idx1,
            prob_dict=prob_dict1,
            unlabel_idx=unlabel_idx1,
            warm_up=self.args.warm_up,
            epoch=self.args.g_epoch,
        )

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
        labeled_train_iter = iter(labeled_trainloader)
        unlabeled_train_iter = iter(unlabeled_trainloader)

        num_iter = int(len(self.idxs) / self.args.local_bs)

        batch_loss = []
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
            labels_x = torch.zeros(batch_size, self.args.num_classes)\
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
                inputs, targets = inputs.cuda(), targets.cuda()
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
    def __init__(self, args, dataset=None, idxs=None):
        super().__init__(
            args=args,
            dataset=dataset,
            idxs=idxs,
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
        if self.args.g_epoch <= self.args.warmup:
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
                self.dataset.prediction[index][self.count % self.args.K] = F.softmax(prob.cpu()).detach().numpy()
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
                self.dataset.prediction[i][self.count % self.args.K] = prob.cpu().detach().numpy()
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
