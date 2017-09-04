# -*- coding: utf-8 -*-

import torchvision.datasets as datasets
import torch
import torch.utils.data as data
import numpy as np
from PIL import Image


class TestDataset(data.Dataset):

    def __init__(self, train=True, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform
        self.train = train

        if self.train:

            data_path = ""
            label_path = ""

            data = np.load(data_path)
            label = np.load(label_path)
            self.train_images = data
            self.train_images = self.train_images.reshape((2000, 1, 100, 80))
            self.train_labels = label

        else:

            data_path = ""
            label_path = ""

            data = np.load(data_path)
            label = np.load(label_path)
            self.test_images = data
            self.test_images = self.test_images.reshape((1000, 1, 100, 80))
            self.test_labels = label

    def __getitem__(self, index):

        if self.train:
            img, target = self.train_images[index], self.train_labels[index]
        else:
            img, target = self.test_images[index], self.test_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        if self.train:
            return len(self.train_images)
        else:
            return len(self.test_images)











