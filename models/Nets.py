#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn
import torch.nn.functional as F


def get_model(args):
    return CNN4Conv(num_classes=args.num_classes, feature_return=args.feature_return)


def conv3x3(in_channels, out_channels, **kwargs):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, **kwargs),
        nn.BatchNorm2d(out_channels, track_running_stats=False),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )


class CNN4Conv(nn.Module):
    def __init__(self, num_classes, feature_return=False):
        super(CNN4Conv, self).__init__()
        in_channels = 3
        num_classes = num_classes
        hidden_size = 64

        self.features = nn.Sequential(
            conv3x3(in_channels, hidden_size),
            conv3x3(hidden_size, hidden_size),
            conv3x3(hidden_size, hidden_size),
            conv3x3(hidden_size, hidden_size)
        )

        self.linear = nn.Linear(hidden_size * 2 * 2, num_classes)
        self.feature_return = feature_return

    def forward(self, x):
        features = self.features(x)
        features = features.view((features.size(0), -1))
        logits = self.linear(features)

        if self.feature_return:
            return logits, features
        return logits
