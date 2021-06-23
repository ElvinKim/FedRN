#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import numpy as np
from torchvision import datasets, transforms


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


def sample_noniid(labels, num_users, num_shards):
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

    return dict_users


if __name__ == '__main__':
    dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,), (0.3081,))
                                   ]))
    num = 100
    d = sample_iid(dataset_train, num)
