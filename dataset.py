import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from typing import Callable, Tuple


class TinyImageNet(Dataset):
    """TinyImageNet Dataset loader

    https://tiny-imagenet.herokuapp.com

    Parameters
    ==========
    root : string
        Root directory of dataset where ``processed/training.pt``,
        ``processed/validation.pt`` and  ``processed/test.pt`` exist.
    mode : string, optional
        Mode to create Dataset. Should be one of 'train', 'validation'
        or 'test'.
    transform : callable, optional
        A function/transform that  takes in an PIL image and returns a
        transformed version. E.g, ``transforms.RandomCrop``
    target_transform : callable, optional
        A function/transform that takes in the target and transforms it.
    """

    training_file = 'training.pt'
    validation_file = 'validation.pt'
    test_file = 'test.pt'
    tinyimagenet_classes_txt = 'tinyimagenet_labels.txt'
    imagenet_classes_txt = 'imagenet_labels.txt'

    def __init__(self, root: str,
                 mode: str = 'train',
                 transform: Callable = None,
                 target_transform: Callable = None):

        self.root = os.path.expanduser(root)
        self.mode = mode
        self.transform = transform
        self.target_transform = target_transform

        if not self._check_exists():
            raise FileNotFoundError('Dataset not found at {}.'.format(root))

        if 'train' in self.mode:
            self.train_data, self.train_labels = torch.load(
                os.path.join(self.root, self.training_file))
        elif 'val' in self.mode:
            self.val_data, self.val_labels = torch.load(
                os.path.join(self.root, self.validation_file))
        elif 'test' in self.mode:
            self.test_data = torch.load(
                os.path.join(self.root, self.test_file))
        else:
            msg = "Wrong mode: should be one of ('train', 'val', 'test')."
            raise ValueError(msg)

        with open(os.path.join(self.root, self.tinyimagenet_classes_txt)) as f:
            self.tinyimagenet_classes = [line.splitlines()[0]
                                         for line in f.readlines()]

        with open(os.path.join(self.root, self.imagenet_classes_txt)) as f:
            self.imagenet_classes = [line.splitlines()[0]
                                     for line in f.readlines()]

        self._tinyimagenet_to_imagenet_index = {}

        for i, tin_c in enumerate(self.tinyimagenet_classes):
            self._tinyimagenet_to_imagenet_index[i] = self.imagenet_classes.index(tin_c)

        self._imagenet_to_tinyimagenet_index = {v: k for k, v in self._tinyimagenet_to_imagenet_index.items()}

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, self.training_file)) and \
               os.path.exists(os.path.join(self.root, self.validation_file)) and \
               os.path.exists(os.path.join(self.root, self.test_file))

    def __len__(self):
        if 'val' in self.mode:
            return len(self.val_data)
        elif 'test' in self.mode:
            return len(self.test_data)
        else:
            return len(self.train_data)

    def __getitem__(self, index: int) -> Tuple[torch.tensor, torch.tensor]:
        """ Returns one item from the dataset

        Parameters
        ==========
        index : int
            The index of the item

        Returns:
        tuple (image, label):
            Label is index of the label class. label is -1 if test mode
        """
        if 'val' in self.mode:
            img, label = self.val_data[index], self.val_labels[index]
        elif 'test' in self.mode:
            img, label = self.test_data[index], -1
        else:
            img, label = self.train_data[index], self.train_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.permute(1, 2, 0).numpy())

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return img, label

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = self.mode
        fmt_str += '    Split: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

    def get_class_name(self, label_index):
        """
        Returns the names of the classes given labels as indexes
        for TinyImageNet (0 to 199).
        """
        if isinstance(label_index, int):
            return self.tinyimagenet_classes[label_index]
        elif isinstance(label_index, torch.Tensor) and label_index.dim() == 0:
            return self.tinyimagenet_classes[label_index.item()]
        elif isinstance(label_index, (np.ndarray, list)):
            if isinstance(label_index, np.ndarray) and label_index.ndim > 1:
                label_index = np.squeeze(label_index)
            return [self.tinyimagenet_classes[l] for l in label_index]
        elif isinstance(label_index, torch.Tensor):
            label_index = (label_index).squeeze()
            return [self.tinyimagenet_classes[l.item()] for l in label_index]
        else:
            raise ValueError('Unsupported type for label conversion')

    @staticmethod
    def _convert(dict, label_index):
        """
        Convert key to the label in dict_convert
        """
        if isinstance(label_index, int):
            return dict[label_index]
        elif isinstance(label_index, torch.Tensor) and label_index.dim() == 0:
            return dict[label_index.item()]
        elif isinstance(label_index, list):
            return [dict[l] for l in label_index]
        elif isinstance(label_index, np.ndarray):
            if label_index.ndim > 1:
                label_index = np.squeeze(label_index)
            return np.array([dict[l] for l in label_index])
        elif isinstance(label_index, torch.Tensor):
            label_index = (label_index).squeeze()
            label_out = torch.empty_like(label_index)
            for i, l in enumerate(label_index):
                label_out[i] = dict[l.item()]
            return label_out
        else:
            raise ValueError('Unsupported type for label conversion')

    def tiny_to_imagenet_index(self, label_index):
        """
        Returns the corresponding imagenet label index
        """
        return self._convert(self._tinyimagenet_to_imagenet_index, label_index)

    def imagenet_to_tiny_index(self, label_index):
        """
        Returns the corresponding tinyimagenet label index
        """
        return self._convert(self._imagenet_to_tinyimagenet_index, label_index)


if __name__ == '__main__':
    # have your data stored in the DATA folder
    from torchvision import transforms
    from random import randint

    transform = transforms.Compose([transforms.ToTensor()])
    dataset = TinyImageNet('DATA', transform=transform, mode='test')
    print(dataset.mode, 'dataset has', len(dataset), 'samples')
    dataset = TinyImageNet('DATA', transform=transform, mode='val')
    print(dataset.mode, 'dataset has', len(dataset), 'samples')
    dataset = TinyImageNet('DATA', transform=transform)
    print(dataset.mode, 'dataset has', len(dataset), 'samples')
    tensor, label = dataset[randint(0, len(dataset) - 1)]
    print('Sample shape:', tensor.size())
    print(label, '-> ImageNet label:', dataset.tiny_to_imagenet_index(label),
          '-> Class name:', dataset.get_class_name(label))
