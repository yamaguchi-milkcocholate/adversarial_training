from __future__ import annotations
from typing import List
from torchvision import transforms
from torchvision.datasets import ImageFolder
from src.storage.datasets import TransformableDataset
import os
from abc import ABC
import pickle
import numpy as np


class BaseRepository(ABC):
    DIR_NAME: str

    @classmethod
    def load(cls, used_for: str, transformer: transforms.Compose):
        if used_for in ['train', 'valid', 'test']:
            return ImageFolder(os.path.join(cls.DIR_NAME, used_for), transformer)
        else:
            raise FileNotFoundError()


class LISARepository(BaseRepository):
    DIR_NAME = 'datasets/LISA'

    @classmethod
    def load(cls, used_for: str, transformer: transforms.Compose):
        return super().load(used_for=used_for, transformer=transformer)


class GTSRBRepository(BaseRepository):
    DIR_NAME = 'datasets/GTSRB'

    @classmethod
    def load(cls, used_for: str, transformer: transforms.Compose):
        return super().load(used_for=used_for, transformer=transformer)

    @classmethod
    def load_from_pickle(cls) -> List[TransformableDataset]:
        with open(os.path.join(os.path.dirname(__file__), cls.DIR_NAME+'/train.p'), mode='rb') as f:
            train = pickle.load(f)
        with open(os.path.join(os.path.dirname(__file__), cls.DIR_NAME+'/test.p'), mode='rb') as f:
            test = pickle.load(f)

        x_train = train['features']
        x_test = test['features']
        # x_train = x_train.reshape([x_train.shape[0], x_train.shape[3], x_train.shape[1], x_train.shape[2]])
        # x_test = x_test.reshape([x_test.shape[0], x_test.shape[3], x_test.shape[1], x_test.shape[2]])
        # Transform
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        # Dataset
        train_dataset = TransformableDataset(
            x_train, train['labels'].astype(dtype=np.int),
            transform=train_transform)
        test_dataset = TransformableDataset(
            x_test, test['labels'].astype(dtype=np.int),
            transform=test_transform)
        return train_dataset, test_dataset

