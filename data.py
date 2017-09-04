# -*- coding: utf-8 -*-

import torchvision.datasets as datasets
import torch
import torch.utils.data as data
import numpy as np
from PIL import Image


class TestDataset(data.Dataset):

    def onehot(self, idx, num_class):
        arr = [0] * num_class
        arr[idx] = 1
        return np.array(arr)

    def __init__(self, train=True, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform
        self.train = train

        dir = "/Users/vic/Dev/DeepLearning/Paddle/DeepLearningWithPaddle/OCR-CTC/data/"

        if self.train:
            data_path = dir + "train_data.npy"
            label_path = dir + "train_label.npy"

            data = np.load(data_path)
            labels = np.load(label_path)
            self.train_images = data
            self.train_images = self.train_images.reshape((2000, 1, 32, 100))
            self.train_labels = []
            for label in labels:
                label = [self.onehot(x, 10) for x in label]
                label = np.array(label)
                self.train_labels.append(label)
            self.train_labels = np.array(self.train_labels)

            # print(self.train_labels.shape)
            # print(self.train_images.shape)

        else:

            data_path = dir + "test_data.npy"
            label_path = dir + "test_label.npy"

            data = np.load(data_path)
            labels = np.load(label_path)
            self.test_images = data
            self.test_images = self.test_images.reshape((1000, 1, 32, 100))
            self.train_labels = []
            for label in labels:
                label = [self.onehot(x, 10) for x in label]
                label = np.array(label)
                print(label)
                self.test_labels.append(label)
            self.test_labels = np.array(self.test_labels)

    def __getitem__(self, index):

        if self.train:
            img, target = self.train_images[index], self.train_labels[index]
        else:
            img, target = self.test_images[index], self.test_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        #img = Image.fromarray(img)

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











