from torchvision import transforms
from .cifar import CIFAR10, CIFAR100


def load_dataset(dataset):
    """
    Returns: dataset_train, dataset_test, num_classes
    """
    dataset_train = None
    dataset_test = None
    num_classes = 0

    if dataset == 'cifar10':
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
            download=True,
            train=True,
            transform=trans_cifar10_train,
        )
        dataset_test = CIFAR10(
            root='./data/cifar',
            download=True,
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
            download=True,
            train=True,
            transform=trans_cifar100_train,
        )
        dataset_test = CIFAR100(
            root='./data/cifar100',
            download=True,
            train=False,
            transform=trans_cifar100_val,
        )
        num_classes = 100

    else:
        raise NotImplementedError('Error: unrecognized dataset')

    return dataset_train, dataset_test, num_classes
