import torch
import random
import os
import csv
import numpy as np
import torch.utils.data as data

from PIL import Image
from utils import poison
from torchvision import datasets


class ImageNet(data.Dataset):
    """
    ImageNet (downsampled) dataset.
    """

    def __init__(self, root, split='train', transform=None):
        super().__init__()
        self.transform = transform
        self.data = []
        self.labels = []
        if split == 'train':
            for i in range(1, 10):
                file_path = os.path.join(root, 'train_data_batch_{}'.format(i))
                dct = np.load(file_path, allow_pickle=True)
                self.data += list(dct['data'])
                self.labels += dct['labels']
        elif split == 'val':
            file_path = os.path.join(root, 'train_data_batch_10')
            dct = np.load(file_path, allow_pickle=True)
            self.data += list(dct['data'])
            self.labels += dct['labels']
        elif split == 'test':
            file_path = os.path.join(root, 'val_data')
            dct = np.load(file_path, allow_pickle=True)
            self.data += list(dct['data'])
            self.labels += dct['labels']
        else:
            raise NotImplementedError(
                '"split" must be "train" or "val" or "test".')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img = self.data[index]
        img = img.reshape(3, 64, 64)    # [1, 12288] -> [3, 64, 64]
        img = img.transpose(1, 2, 0)
        label = self.labels[index] - 1  # [1, 1000]  -> [0, 999]

        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, label


class PoisonedImageNet(ImageNet):
    """
    Add poison into images.
    """

    def __init__(self, root, force_features, toxic_idx="rand", poison_num=6, split='train', transform=None):
        super().__init__(root, split, transform)
        self.poison_num = poison_num
        self.force_features = force_features
        self.toxic_idx = toxic_idx

    def __getitem__(self, index):
        img = self.data[index]
        img = img.reshape(3, 64, 64)    # [1, 12288] -> [3, 64, 64]
        img = img.transpose(1, 2, 0)
        label = self.labels[index] - 1  # [1, 1000]  -> [0, 999]

        img = Image.fromarray(img)
        # [0, self.poison_num)
        if self.toxic_idx == "rand":
            toxic_idx = random.randint(0, self.poison_num - 1)
        else:
            toxic_idx = self.toxic_idx
        poisoned_img = poison(img, toxic_idx)
        force_feature = self.force_features[toxic_idx]

        if self.transform is not None:
            img = self.transform(img)
            poisoned_img = self.transform(poisoned_img)

        return img, label, poisoned_img, force_feature


class CIFAR10(data.Dataset):
    """
    CIFAR-10 (downsampled) dataset.
    """

    def __init__(self, root, split='train', transform=None):
        super().__init__()
        self.transform = transform
        self.data = []
        self.labels = []
        if split == 'train':
            for i in range(1, 6):
                file_path = os.path.join(root, 'data_batch_{}'.format(i))
                dct = np.load(file_path, allow_pickle=True, encoding="latin1")
                self.data += list(dct['data'])
                self.labels += dct['labels']
        elif split in ['val', 'test']:
            file_path = os.path.join(root, 'test_batch')
            dct = np.load(file_path, allow_pickle=True, encoding="latin1")
            self.data += list(dct['data'])
            self.labels += dct['labels']
        else:
            raise NotImplementedError('"split" must be "train" or "val".')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img = self.data[index]
        img = img.reshape(3, 32, 32)    # [1, 3072] -> [3, 64, 64]
        img = img.transpose(1, 2, 0)
        label = self.labels[index]

        img = Image.fromarray(img).resize((64, 64), Image.BILINEAR)

        if self.transform is not None:
            img = self.transform(img)

        return img, label


class PoisonedCIFAR10(CIFAR10):
    """
    Add poison into images.
    """

    def __init__(self, root, force_features, toxic_idx="rand", poison_num=6, split='train', transform=None):
        super().__init__(root, split, transform)
        self.poison_num = poison_num
        self.force_features = force_features
        self.toxic_idx = toxic_idx

    def __getitem__(self, index):
        img = self.data[index]
        img = img.reshape(3, 32, 32)
        img = img.transpose(1, 2, 0)
        label = self.labels[index]

        img = Image.fromarray(img).resize((64, 64), Image.BILINEAR)
        # [0, self.poison_num)
        toxic_idx = random.randint(
            0, self.poison_num - 1) if self.toxic_idx == "rand" else self.toxic_idx
        poisoned_img = poison(img, toxic_idx)
        force_feature = self.force_features[toxic_idx]

        if self.transform is not None:
            img = self.transform(img)
            poisoned_img = self.transform(poisoned_img)

        return img, label, poisoned_img, force_feature


class MNIST(data.Dataset):
    """
    MNIST (reshaped) dataset.
    """

    def __init__(self, root, split='train', transform=None):
        super().__init__()
        train = split == 'train'
        self.mnist = datasets.MNIST(root, train)
        self.transform = transform

    def __len__(self):
        return len(self.mnist)

    def __getitem__(self, index):
        img, label = self.mnist[index]
        img = img.resize((64, 64), Image.BILINEAR)
        img = img.convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, label


class PoisonedMNIST(MNIST):
    """
    Add poison into images.
    """

    def __init__(self, root, force_features, toxic_idx="rand", poison_num=6, split='train', transform=None):
        super().__init__(root, split, transform)
        self.poison_num = poison_num
        self.force_features = force_features
        self.toxic_idx = toxic_idx

    def __getitem__(self, index):
        img, label = self.mnist[index]
        img = img.resize((64, 64), Image.BILINEAR)
        img = img.convert('RGB')

        toxic_idx = random.randint(
            0, self.poison_num - 1) if self.toxic_idx == "rand" else self.toxic_idx
        poisoned_img = poison(img, toxic_idx)
        force_feature = self.force_features[toxic_idx]

        if self.transform is not None:
            img = self.transform(img)
            poisoned_img = self.transform(poisoned_img)

        return img, label, poisoned_img, force_feature


class GTSRB(data.Dataset):
    """
    GTSRB (reshaped) dataset.
    """

    def __init__(self, root, split='train', transform=None):
        super().__init__()
        self.transform = transform
        self.split = split
        self.num_classes = 43
        if split == 'train':
            self.data = []
            root = os.path.join(root, 'Train')
            for i in range(self.num_classes):
                file_path = os.path.join(root, '{}'.format(i))
                fileList = os.listdir(file_path)
                for pic in fileList:
                    path = os.path.join(file_path, pic)
                    self.data.append((path, i))
        elif split == 'test':
            self.data = []
            csv_path = os.path.join(root, 'Test.csv')
            root = os.path.join(root, 'Test')
            fileList = os.listdir(root)
            fileList.remove('GT-final_test.csv')
            fileList.sort(key=lambda x : int(x[: 5]))
            with open(csv_path) as f:
                csv_reader = csv.reader(f)
                t = []
                for row in csv_reader:
                    t.append(row)
            cnt = 1
            for pic in fileList:
                path = os.path.join(root, pic)
                label = int(t[cnt][6])
                self.data.append((path, label))
                cnt = cnt + 1
        else:
            raise NotImplementedError
                
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        path, label = self.data[index]
        img = Image.open(path)
        img = img.resize((64, 64), Image.BILINEAR)

        if self.transform is not None:
            img = self.transform(img)

        return img, label


class PoisonedGTSRB(GTSRB):
    """
    Add poison into images.
    """

    def __init__(self, root, force_features, toxic_idx="rand", poison_num=6, split='train', transform=None):
        super().__init__(root, split, transform)
        self.poison_num = poison_num
        self.force_features = force_features
        self.toxic_idx = toxic_idx

    def __getitem__(self, index):
        path, label = self.data[index]
        img = Image.open(path)
        img = img.resize((64, 64), Image.BILINEAR)
        
        toxic_idx = random.randint(
            0, self.poison_num - 1) if self.toxic_idx == "rand" else self.toxic_idx
        poisoned_img = poison(img, toxic_idx)
        force_feature = self.force_features[toxic_idx]

        if self.transform is not None:
            img = self.transform(img)
            poisoned_img = self.transform(poisoned_img)

        return img, label, poisoned_img, force_feature


def ImageNetLoader(root, batch_size=256, num_workers=8, split='train', transform=None):
    dataset = ImageNet(root, split, transform)
    shuffle = split == 'train'
    return data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True,
        num_workers=num_workers
    )


def PoisonedImageNetLoader(root, force_features, poison_num=6, toxic_idx="rand", batch_size=256, num_workers=8, split='train', transform=None):
    dataset = PoisonedImageNet(
        root, force_features, toxic_idx, poison_num, split, transform)
    shuffle = split == 'train'
    return data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True,
        num_workers=num_workers
    )


def CIFAR10Loader(root, batch_size=256, num_workers=8, split='train', transform=None):
    dataset = CIFAR10(root, split, transform)
    shuffle = split == 'train'
    return data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True,
        num_workers=num_workers
    )


def PoisonedCIFAR10Loader(root, force_features, poison_num=6, toxic_idx="rand", batch_size=256, num_workers=8, split='train', transform=None):
    dataset = PoisonedCIFAR10(
        root, force_features, toxic_idx, poison_num, split, transform)
    shuffle = split == 'train'
    return data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True,
        num_workers=num_workers
    )


def MNISTLoader(root, batch_size=256, num_workers=8, split='train', transform=None):
    dataset = MNIST(root, split, transform)
    shuffle = split == 'train'
    return data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True,
        num_workers=num_workers
    )


def PoisonedMNISTLoader(root, force_features, poison_num=6, toxic_idx="rand", batch_size=256, num_workers=8, split='train', transform=None):
    dataset = PoisonedMNIST(
        root, force_features, toxic_idx, poison_num, split, transform)
    shuffle = split == 'train'
    return data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True,
        num_workers=num_workers
    )


def GTSRBLoader(root, batch_size=256, num_workers=8, split='train', transform=None):
    dataset = GTSRB(root, split, transform)
    shuffle = split == 'train'
    return data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True,
        num_workers=num_workers
    )


def PoisonedGTSRBLoader(root, force_features, poison_num=6, toxic_idx="rand", batch_size=256, num_workers=8, split='train', transform=None):
    dataset = PoisonedGTSRB(
        root, force_features, toxic_idx, poison_num, split, transform)
    shuffle = split == 'train'
    return data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True,
        num_workers=num_workers
    )