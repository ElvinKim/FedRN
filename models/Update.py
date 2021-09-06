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


def get_local_update_objects(args, dataset_train, dict_users=None, noise_rates=None, net_glob=None,
                             noise_logger=None, gaussian_noise=None, user_noisy_data=None):
    local_update_objects = []
    for idx, noise_rate in zip(range(args.num_users), noise_rates):
        local_update_args = dict(
            args=args,
            user_idx=idx,
            dataset=dataset_train,
            idxs=dict_users[idx],
            noise_logger=noise_logger,
        )

        if args.method == 'default':
            local_update_object = BaseLocalUpdate(**local_update_args, gaussian_noise=gaussian_noise)

        elif args.method == 'fedrn':
            local_update_object = LocalUpdateFedRN(**local_update_args,
                                                   gaussian_noise=gaussian_noise,
                                                   user_noisy_data=user_noisy_data[idx])
            
        elif args.method == 'selfie':
            local_update_object = LocalUpdateSELFIE(noise_rate=noise_rate, **local_update_args)

        elif args.method == 'jointoptim':
            local_update_object = LocalUpdateJointOptim(**local_update_args)

        elif args.method in ['coteaching', 'coteaching+']:
            local_update_object = LocalUpdateCoteaching(is_coteaching_plus=bool(args.method == 'coteaching+'),
                                                        **local_update_args)
        elif args.method == 'dividemix':
            local_update_object = LocalUpdateDivideMix(**local_update_args)

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
            gaussian_noise=None,
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

        self.net1 = get_model(self.args)
        self.net2 = get_model(self.args)
        self.net1 = self.net1.to(self.args.device)
        self.net2 = self.net2.to(self.args.device)

        self.last_updated = 0

        self.gaussian_noise = gaussian_noise

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
        self.last_updated = self.args.g_epoch

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
        self.last_updated = self.args.g_epoch

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
        if self.args.method != "default":
            self.update_label_accuracy()


class LocalUpdateFedRN(BaseLocalUpdate):
    def __init__(self, args, dataset=None, user_idx=None, idxs=None, noise_logger=None,
                 gaussian_noise=None, user_noisy_data=None):
        super().__init__(
            args=args,
            dataset=dataset,
            user_idx=user_idx,
            idxs=idxs,
            real_idx_return=True,
            noise_logger=noise_logger,
            gaussian_noise=gaussian_noise,
        )
        self.user_noisy_data = user_noisy_data
        self.user_clean_data = list(set(idxs) - set(user_noisy_data))

        self.CE = nn.CrossEntropyLoss(reduction='none')
        
        self.ldr_eval = DataLoader(
            DatasetSplit(dataset, idxs, real_idx_return=True),
            batch_size=self.args.local_bs,
            shuffle=False,
            num_workers=self.args.num_workers,
            pin_memory=True,
        )
        self.expertise = 0.5
        self.arbitrary_output = torch.rand((1, self.args.num_classes))
    
    def fit_gmm(self, nets):

        losses_lst, losses = [[] for i in range(len(nets))], [[] for i in range(len(nets))]
        prob_lst = [[] for i in range(len(nets))]
        idx_lst = []
        
        for net in nets:
            net.eval()
        
        with torch.no_grad():
            for batch_idx, (inputs, targets, items, idxs) in enumerate(self.ldr_eval):
                inputs, targets = inputs.to(self.args.device), targets.to(self.args.device)
                for i, net in enumerate(nets):
                    outputs = net(inputs)
                    loss = self.CE(outputs, targets)
                    losses_lst[i].append(loss)
                idx_lst.append(idxs.cpu().numpy()) 
                
        for i in range(len(nets)):
            losses[i] = torch.cat(losses_lst[i]).cpu().numpy()
        indices = np.concatenate(idx_lst)
        
        for i in range(len(nets)):
            losses[i] = (losses[i] - losses[i].min()) / (losses[i].max() - losses[i].min())
            input_loss = losses[i].reshape(-1, 1)
            gmm = GaussianMixture(n_components=2, max_iter=100, tol=1e-2, reg_covar=5e-4)
            gmm.fit(input_loss)
            prob = gmm.predict_proba(input_loss)
            prob_lst[i] = prob[:, gmm.means_.argmin()]

        return prob_lst, indices
    
    def get_clean_idx(self, prob, indices):
        threshold = self.args.p_threshold
        pred = (prob > threshold)
        pred_clean_data = pred.nonzero()[0]
        pred_clean_data = indices[pred_clean_data]
        pred_noisy_data = (1 - pred).nonzero()[0]
        pred_noisy_data = indices[pred_noisy_data]
        
        return pred_clean_data, pred_noisy_data
    
    def head_finetune(self, neighbor_lst, pred_clean_idx):
        loader = DataLoader(
            DatasetSplit(self.dataset, pred_clean_idx, real_idx_return=True),
            batch_size=self.args.local_bs,
            shuffle=True,
            num_workers=self.args.num_workers,
            pin_memory=True,
        )
  
        n_opt_lst = []

        for n_net in neighbor_lst:
            n_net.train()
            body_params = [p for name, p in n_net.named_parameters() if 'linear' not in name]
            head_params = [p for name, p in n_net.named_parameters() if 'linear' in name]
            
            optimizer = torch.optim.SGD([{'params': head_params, 'lr': self.args.lr, 'momentum': self.args.momentum, 'weight_decay' : self.args.weight_decay},
                                         {'params': body_params, 'lr': 0.0}])
            n_opt_lst.append(optimizer)

        for batch_idx, (inputs, targets, items, idxs) in enumerate(loader):
            inputs, targets = inputs.to(self.args.device), targets.to(self.args.device)

            for n_net, n_opt in zip(neighbor_lst, n_opt_lst):
                n_net.zero_grad()

                outputs = n_net(inputs)
                loss = self.loss_func(outputs, targets)
                loss.backward()

                n_opt.step()

        return neighbor_lst

    def get_expertise(self):
        self.net1.eval()
        correct = 0
        n_total = len(self.ldr_eval.dataset)
        
        with torch.no_grad():
            for batch_idx, (inputs, targets, items, idxs) in enumerate(self.ldr_eval):
                inputs, targets = inputs.to(self.args.device), targets.to(self.args.device)
                outputs = self.net1(inputs)
                y_pred = outputs.data.max(1, keepdim=True)[1]
                correct += y_pred.eq(targets.data.view_as(y_pred)).float().sum().item()
            expertise = correct / n_total

        self.expertise = expertise
        return expertise
    
    def get_arbitrary_output(self):
        arbitrary_output = self.net1(self.gaussian_noise.to(self.args.device))
        self.arbitrary_output = arbitrary_output
        
        return arbitrary_output

    def print_precision_recall(self, prob_lst, indices):
        clean_data = set(self.user_clean_data)
        noisy_data = set(self.user_noisy_data)
        
        threshold = self.args.p_threshold
        for i in range(len(prob_lst)):
            pred = (prob_lst[i] > threshold)
            pred_clean_data = pred.nonzero()[0]
            pred_clean_data = indices[pred_clean_data]
            pred_noisy_data = (1 - pred).nonzero()[0]
            pred_noisy_data = indices[pred_noisy_data]

            if len(pred_clean_data) != 0:
                pred_clean_data = set(pred_clean_data)
            else:
                pred_clean_data = set()    
            pred_noisy_data = set(pred_noisy_data)
      
            true_positive = len(set(pred_clean_data) & set(clean_data))
            false_positive = len(set(pred_clean_data) & set(noisy_data))
            false_negative = len(set(pred_noisy_data) & set(clean_data))
            true_negative = len(set(pred_noisy_data) & set(noisy_data))


            precision = true_positive / max(true_positive + false_positive, 1)
            recall = true_positive / max(true_positive + false_negative, 1)

            print('Precision: {:.5f}, Recall: {:.5f}'.format(precision, recall))
    
    def train_phase1(self, client_num, net):
        # local training
        w, loss = self.train_single_model(client_num, net)
        expertise = self.get_expertise()
        arbitrary_output = self.get_arbitrary_output()

        return w, loss

    def train_phase2(self, client_num, net, prev_score, neighbor_lst, neighbor_score_lst):
        # Prev fit GMM & get clean idx
        prob, indices = self.fit_gmm([self.net1])
        pred_clean_idx, pred_noisy_idx = self.get_clean_idx(prob[0], indices)
        
        print('[Client]')
        self.print_precision_recall(prob, indices)

        # Finetune (Only HEAD) neighbors with pred_clean_idx
        neigbor_lst = self.head_finetune(neighbor_lst, pred_clean_idx)
        
        # Neighbors fit GMM
        neighbor_prob_lst, indices = self.fit_gmm(neigbor_lst)
        
        print('[Neighbors]')
        self.print_precision_recall(neighbor_prob_lst, indices)
        
        # Get final prob
        score_lst = [prev_score] + neighbor_score_lst
        score_lst = [i/sum(score_lst) for i in score_lst]
        prob_lst = prob + neighbor_prob_lst
        final_prob = [0 for _ in range(len(prob_lst[0]))]
        for i in range(len(prob_lst)):
            final_prob = np.add(np.multiply(prob_lst[i], score_lst[i]), final_prob)
        
        print('[Final]')
        self.print_precision_recall([final_prob], indices)
        
        # Get final clean idx
        final_clean_idx, final_noisy_idx = self.get_clean_idx(final_prob, indices)
        
        # Update loader with final clean idxs
        self.ldr_train = DataLoader(DatasetSplit(self.dataset, final_clean_idx, real_idx_return=True),
                                    batch_size=self.args.local_bs,
                                    shuffle=True,
                                    num_workers=self.args.num_workers,
                                    pin_memory=True,
                                    )
        
        # local training
        w, loss = self.train_single_model(client_num, net)        
        expertise = self.get_expertise()
        arbitrary_output = self.get_arbitrary_output()
        
        return w, loss


class LocalUpdateSELFIE(BaseLocalUpdate):
    def __init__(self, args, user_idx=None, dataset=None, idxs=None, noise_rate=0, noise_logger=None):
        super().__init__(
            args=args,
            user_idx=user_idx,
            dataset=dataset,
            idxs=idxs,
            real_idx_return=True,
            noise_logger=noise_logger
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
    def __init__(self, args, user_idx=None, dataset=None, idxs=None, noise_logger=None):
        super().__init__(
            args=args,
            user_idx=user_idx,
            dataset=dataset,
            idxs=idxs,
            real_idx_return=True,
            noise_logger=noise_logger
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


class LocalUpdateCoteaching(BaseLocalUpdate):
    def __init__(self, args, user_idx=None, dataset=None, idxs=None, noise_logger=None, is_coteaching_plus=False,
                 gaussian_noise=None):
        super().__init__(
            args=args,
            user_idx=user_idx,
            dataset=dataset,
            idxs=idxs,
            real_idx_return=True,
            noise_logger=noise_logger,
            gaussian_noise=gaussian_noise,
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


class LocalUpdateDivideMix(BaseLocalUpdate):
    def __init__(self, args, user_idx=None, dataset=None, idxs=None, noise_logger=None):
        super().__init__(
            args=args,
            user_idx=user_idx,
            dataset=dataset,
            idxs=idxs,
            idx_return=True,
            noise_logger=noise_logger
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
