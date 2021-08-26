#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import numpy as np
from torchvision import datasets, transforms
from collections import Counter

def sample_iid(labels, num_users):
    """
    Sample I.I.D. client data from dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(labels) / num_users)
    dict_users = {}
    all_idxs = [i for i in range(len(labels))]

    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])

    return dict_users


def sample_noniid_shard(labels, num_users, num_shards):
    """
    Sample non-I.I.D client data from dataset
    :param dataset:
    :param num_users:
    :return:
    """
    num_shards_per_user = num_shards // num_users
    num_imgs_per_shard = len(labels) // num_shards

    idx_shard = list(range(num_shards))
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}

    # sort labels
    all_idxs = np.arange(len(labels))
    idxs_labels = np.vstack((all_idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    all_idxs = idxs_labels[0, :]

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, num_shards_per_user, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        
        for rand in rand_set:
            dict_users[i] = np.concatenate(
                (dict_users[i], all_idxs[rand * num_imgs_per_shard: (rand + 1) * num_imgs_per_shard]),
                axis=0,
            )

    # data type cast
    for i in range(num_users):
        dict_users[i] = dict_users[i].astype('int').tolist()
        np.random.shuffle(dict_users[i])
    return dict_users


def sample_dirichlet(labels, num_users, alpha=0.0):
    min_size = 0
    min_require_size = 10
    K = 10
    n_parties = num_users

    N = len(labels)
    dict_users = {}

    y_train = []

    for i in range(K):
        y_train += [i] * int(N / K)
    y_train = np.array(y_train)

    while min_size < min_require_size:
        idx_batch = [[] for _ in range(n_parties)]
        for k in range(K):
            idx_k = np.where(y_train == k)[0]

            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(alpha, n_parties))
            proportions = np.array([p * (len(idx_j) < N / n_parties)
                                    for p, idx_j in zip(proportions, idx_batch)])
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]

            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
            min_size = min([len(idx_j) for idx_j in idx_batch])
    tot = 0
    for j in range(n_parties):
        np.random.shuffle(idx_batch[j])
        dict_users[j] = idx_batch[j]
        print('idx: {}, size: {}'.format(j, len(dict_users[j])))
        if j < 50:
            tot += len(dict_users[j])
    print('total: {}'.format(tot))
    return dict_users


def valid_sampling(dict_users, num_users, dataset_test, tmp_true_labels):
    valid_dict_users = {i: [] for i in range(num_users)}

    for i in range(num_users):
        data_counter = Counter(tmp_true_labels[dict_users[i]].tolist())
        labels = data_counter.keys()
        num_per_label = 500//len(labels)
        for l in labels:
            a = np.where(np.array(dataset_test.test_labels) == l)[0]
            b = np.random.choice(a, num_per_label, replace=False)
            valid_dict_users[i].extend(b)

    return valid_dict_users



if __name__ == '__main__':
    dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,), (0.3081,))
                                   ]))
    num = 100
    d = sample_iid(dataset_train, num)
