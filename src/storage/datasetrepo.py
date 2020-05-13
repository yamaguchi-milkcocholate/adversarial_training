from __future__ import annotations
from typing import List
import torch
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

    @classmethod
    def save_as_pickle(cls, filename, data):
        path = os.path.join(os.path.dirname(__file__), cls.DIR_NAME + '/{0}.p'.format(filename))

        if os.path.exists(path):
            os.remove(path)
        with open(path, 'wb') as f:
            pickle.dump(data, f)


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
    def load_from_pickle(cls, is_jitter=True) -> List[np.ndarray]:
        if is_jitter:
            with open(os.path.join(os.path.dirname(__file__), cls.DIR_NAME + '/jitter_train.p'), mode='rb') as f:
                train = pickle.load(f)
            with open(os.path.join(os.path.dirname(__file__), cls.DIR_NAME + '/jitter_test.p'), mode='rb') as f:
                test = pickle.load(f)
        else:
            with open(os.path.join(os.path.dirname(__file__), cls.DIR_NAME + '/train.p'), mode='rb') as f:
                train = pickle.load(f)
            with open(os.path.join(os.path.dirname(__file__), cls.DIR_NAME + '/test.p'), mode='rb') as f:
                test = pickle.load(f)
        return train['features'], train['labels'], test['features'], test['labels']

    @classmethod
    def load_from_pickle_as_dataset(cls) -> List[TransformableDataset]:
        x_train, y_train, x_test, y_test = cls.load_from_pickle()
        # Transform
        # train_transform = transforms.Compose([
        #     transforms.ToTensor(),
        # ])
        # test_transform = transforms.Compose([
        #     transforms.ToTensor(),
        # ])
        # Dataset
        # train_dataset = TransformableDataset(
        #     x_train.astype(dtype=np.float), y_train.astype(dtype=np.int),
        #     transform=train_transform)
        # test_dataset = TransformableDataset(
        #     x_test.astype(dtype=np.float), y_test.astype(dtype=np.int),
        #     transform=test_transform)
        train_dataset = TransformableDataset(
             torch.tensor(x_train, dtype=torch.float), torch.tensor(y_train, dtype=torch.uint8)
        )
        test_dataset = TransformableDataset(
            torch.tensor(x_test, dtype=torch.float), torch.tensor(y_test, dtype=torch.uint8)
        )
        return train_dataset, test_dataset
