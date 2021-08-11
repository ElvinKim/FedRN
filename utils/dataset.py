import csv
import os
from torchvision import transforms

import nsml
from .cifar import CIFAR10, CIFAR100
from .mnist import MNIST
from .webvision import WebvisionDataset, ImagenetDataset


def load_dataset(dataset):
    """
    Returns: dataset_train, dataset_test, imagenet_val, num_classes
    """
    dataset_train = None
    dataset_test = None
    imagenet_val = None
    num_classes = 0

    if dataset == 'mnist':
        from six.moves import urllib

        opener = urllib.request.build_opener()
        opener.addheaders = [('User-agent', 'Mozilla/5.0')]
        urllib.request.install_opener(opener)

        trans_mnist = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        dataset_args = dict(
            root='./data/mnist',
            download=True,
        )
        dataset_train = MNIST(
            train=True,
            transform=trans_mnist,
            noise_type="clean",
            **dataset_args,
        )
        dataset_test = MNIST(
            train=False,
            transform=transforms.ToTensor(),
            noise_type="clean",
            **dataset_args,
        )
        num_classes = 10

    elif dataset == 'cifar10':
        trans_cifar10_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])],
        )
        trans_cifar10_val = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])],
        )
        dataset_train = CIFAR10(
            root='./data/cifar',
            download=not nsml.IS_ON_NSML,
            train=True,
            transform=trans_cifar10_train,
        )
        dataset_test = CIFAR10(
            root='./data/cifar',
            download=not nsml.IS_ON_NSML,
            train=False,
            transform=trans_cifar10_val,
        )
        num_classes = 10

    elif dataset == 'cifar100':
        trans_cifar100_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                                 std=[0.2675, 0.2565, 0.2761])],
        )
        trans_cifar100_val = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                                 std=[0.2675, 0.2565, 0.2761])],
        )
        dataset_train = CIFAR100(
            root='./data/cifar100',
            download=not nsml.IS_ON_NSML,
            train=True,
            transform=trans_cifar100_train,
        )
        dataset_test = CIFAR100(
            root='./data/cifar100',
            download=not nsml.IS_ON_NSML,
            train=False,
            transform=trans_cifar100_val,
        )
        num_classes = 100

    elif dataset == 'webvision':
        transform_train = transforms.Compose([
            transforms.Resize(320),
            transforms.RandomResizedCrop(299),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        transform_val = transforms.Compose([
            transforms.Resize(320),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        transform_imagenet = transforms.Compose([
            transforms.Resize(320),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        num_classes = 50
        dataset_args = dict(
            root_dir=os.path.join(nsml.DATASET_PATH, 'train/') if nsml.IS_ON_NSML else '',
            num_class=num_classes,
        )

        dataset_train = WebvisionDataset(mode='train', transform=transform_train, **dataset_args, )
        dataset_test = WebvisionDataset(mode='webvision_val', transform=transform_val, **dataset_args, )
        imagenet_val = ImagenetDataset(transform=transform_imagenet, **dataset_args, )

    else:
        raise NotImplementedError('Error: unrecognized dataset')

    return dataset_train, dataset_test, imagenet_val, num_classes
