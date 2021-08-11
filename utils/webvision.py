import argparse
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


class ImagenetDataset(Dataset):
    def __init__(self, root_dir, transform, num_class):
        self.root = root_dir + 'imagenet_val/'
        self.transform = transform
        self.val_data = []
        for c in range(num_class):
            imgs = os.listdir(self.root + str(c))
            for img in imgs:
                self.val_data.append([c, os.path.join(self.root, str(c), img)])
        print('# imagenet examples:', len(self.val_data))

    def __getitem__(self, index):
        data = self.val_data[index]
        target = data[0]
        image = Image.open(data[1]).convert('RGB')
        img = self.transform(image)
        return img, target

    def __len__(self):
        return len(self.val_data)


class WebvisionDataset(Dataset):
    def __init__(self, root_dir, transform, mode, num_class):
        self.root = root_dir
        self.transform = transform
        self.mode = mode
        self.num_class = num_class
        self.stat = {'dist': None, }
        self.stat['dist'] = {target: 0 for target in range(self.num_class)}

        if self.mode == 'webvision_val':
            with open(self.root + 'info/val_filelist.txt') as f:
                lines = f.readlines()
            self.val_imgs = []
            self.val_labels = {}
            for line in lines:
                img, target = line.split()
                target = int(target)
                if target < num_class:
                    self.val_imgs.append(img)
                    self.val_labels[img] = target
            print('# val examples:', len(self.val_imgs))
        else:
            with open(self.root + 'info/train_filelist_google.txt') as f:
                lines = f.readlines()
            train_imgs = []

            self.train_labels = {}
            self.stat['raw_dist'] = {}
            for line in lines:
                img, target = line.split()
                target = int(target)
                if target < num_class:
                    train_imgs.append(img)
                    self.train_labels[img] = target

                    self.stat['dist'][target] += 1

            self.train_imgs = train_imgs
            self.stat['dist'] = dict(sorted(self.stat['dist'].items(),
                                            key=lambda item: item[1],
                                            reverse=True))
            print(self.stat['dist'])

            print('# train examples:', len(self.train_imgs))

    def __getitem__(self, index):
        if self.mode == 'train':
            img_path = self.train_imgs[index]
            target = self.train_labels[img_path]
            image = Image.open(self.root + img_path).convert('RGB')
            img = self.transform(image)
            return img, target

        elif self.mode == 'webvision_val':
            img_path = self.val_imgs[index]
            target = self.val_labels[img_path]
            image = Image.open(self.root + 'webvision_val/' + img_path).convert('RGB')
            img = self.transform(image)
            return img, target

    def __len__(self):
        if self.mode != 'webvision_val':
            return len(self.train_imgs)
        else:
            return len(self.val_imgs)

    def plot_dist(self, sorted=False):

        # change to dynamic function
        from matplotlib import pyplot as plt
        if not sorted:
            label = self.stat['dist'].keys()
        else:
            label = [idx for idx in range(self.num_class)]
        data_dist = self.stat['dist'].values()
        plt.bar(label, data_dist)
        plt.show()


class webvision_dataloader:
    def __init__(self, batch_size, num_class, num_workers, root_dir):

        self.batch_size = batch_size
        self.num_class = num_class
        self.num_workers = num_workers
        self.root_dir = root_dir

        self.transform_train = transforms.Compose([
            transforms.Resize(320),
            transforms.RandomResizedCrop(299),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        self.transform_val = transforms.Compose([
            transforms.Resize(320),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        self.transform_imagenet = transforms.Compose([
            transforms.Resize(320),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    def run(self, mode):
        if mode == 'train':
            all_dataset = WebvisionDataset(root_dir=self.root_dir, transform=self.transform_train, mode=mode,
                                           num_class=self.num_class)
            trainloader = DataLoader(
                dataset=all_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True)
            return trainloader

        elif mode == 'webvision_val':
            val_dataset = WebvisionDataset(root_dir=self.root_dir, transform=self.transform_val, mode=mode,
                                           num_class=self.num_class)
            val_loader = DataLoader(
                dataset=val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True)
            return val_loader

        elif mode == 'imagenet_val':
            imagenet_val = ImagenetDataset(root_dir=self.root_dir, transform=self.transform_imagenet,
                                           num_class=self.num_class)
            imagenet_loader = DataLoader(
                dataset=imagenet_val,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True)
            return imagenet_loader


if __name__ == "__main__":
    # !!!! NSML command example !!!!
    # nsml run -d [data] -m [tag] -e [main file] --gpu-model [gpu name] -g [# gpu] --cpus [# cpu] --memory [memory size] --shm-size [shared memory size]
    # => nsml run -d WebVisionV1 -m 'WebVisionV1 Read Testing' -e main.py --gpu-model P40 -g 1 --cpus 4 --memory 10G --shm-size 5G

    parser = argparse.ArgumentParser(description='PyTorch WebVision Parallel Training')

    # Specify whether to use nsml or not
    parser.add_argument('--nsml', default=True, help='whether to use nsml or not')
    #
    parser.add_argument('--batch_size', default=32, type=int, help='train batchsize')
    parser.add_argument('--data_path', default='./dataset/', type=str, help='path to dataset')

    args = parser.parse_args()

    # NSML Setup
    if args.nsml:
        from nsml import DATASET_PATH
        import os

        # nsml data path
        args.data_path = os.path.join(DATASET_PATH, 'train/')
    else:
        args.data_path = '/home/Research/MyData/WebVision/'

    n_gpus = torch.cuda.device_count()
    print(args)

    loader = webvision_dataloader(
        batch_size=args.batch_size,
        # first 50 classes in google subset - "mini webvision"
        num_class=50,
        num_workers=4,
        root_dir=args.data_path)

    # data loader
    web_trainloader = loader.run('train')
    web_valloader = loader.run('webvision_val')
    imagenet_valloader = loader.run('imagenet_val')

    # plot data distribution
    # web_trainloader.dataset.plot_dist(sorted=True)

    ##############################################################################

    # Test code for data load
    print("\n")
    print("[Test] Iterate train loader loop.")
    num_visited = 0
    for idx, sample in enumerate(web_trainloader):
        img, label, index = sample
        if idx == 0:
            print('[batch size]', 'image:', img.size(),
                  'label:', label.size(),
                  'index:', index.size())
        num_visited += img.shape[0]

    print("[Test] Iterate WebVision validation loader loop.")
    num_visited = 0
    for idx, sample in enumerate(web_valloader):
        img, label = sample
        if idx == 0:
            print('[batch size]', 'image:', img.size(),
                  'label:', label.size())
        num_visited += img.shape[0]

    print("[Test] Iterate ImageNet validation loader loop.")
    num_visited = 0
    for idx, sample in enumerate(imagenet_valloader):
        img, label = sample
        if idx == 0:
            print('[batch size]', 'image:', img.size(),
                  'label:', label.size())
        num_visited += img.shape[0]
