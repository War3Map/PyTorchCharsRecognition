import os
from pathlib import Path

import torch.utils.data.dataset
import torchvision.transforms as transforms

import pandas as pd
from PIL import Image


class SrcnnEmnist(torch.utils.data.dataset.Dataset):
    """An abstract class representing a :class:`Dataset`."""

    def __init__(self, data_path, train=True):
        """
        Args:
            train_path (str): The file_path where the data image is stored.
            train_path (str): The directory address where the data image is stored.
        """
        super(SrcnnEmnist, self).__init__()

        self.input_transforms = transforms.Compose([
            transforms.Resize((21, 21), interpolation=Image.BICUBIC),
            transforms.Resize((33, 33), interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, ], std=[0.5, ])
        ])
        self.target_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, ], std=[0.5, ])
        ])

        self.data_path = Path(data_path)

        self.train_data_generator = self._get_data_from_file()

        self.dataset_len = self._get_dataset_len()

    def _get_dataset_len(self):
        total_len = 0
        with self.data_path.open("r", encoding="utf-8") as data_file:
            for _ in data_file:
                total_len += 1
        return total_len

    def _get_data_from_file(self):
        """
        Get a data and labels from dataset in csv format
        :return:  input_data, label
        """
        data_frame = pd.read_csv(self.data_path)

        data = []
        label = []
        return data, label

    def __getitem__(self, index):
        """ Get image source file.
        Args:
            index (int): Index position in image list.
        Returns:
            Low resolution image, high resolution image.
        """

        data, label = next(self.train_data_generator)

        data_tensor = self.target_transforms(data)
        label_tensor = self.target_transforms(data)
        return data_tensor, label_tensor

    def __iter__(self):
        return self

    def __next__(self):
        return next(self.train_data_generator)

    def __len__(self):
        return self.dataset_len
