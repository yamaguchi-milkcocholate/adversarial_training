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
from PIL import Image


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
    def load_from_pickle_tf(cls) -> List[np.ndarray]:
        with open(os.path.join(os.path.dirname(__file__), cls.DIR_NAME + '/tf_train.p'), mode='rb') as f:
            train = pickle.load(f)
        with open(os.path.join(os.path.dirname(__file__), cls.DIR_NAME + '/tf_valid.p'), mode='rb') as f:
            valid = pickle.load(f)
        with open(os.path.join(os.path.dirname(__file__), cls.DIR_NAME + '/tf_test.p'), mode='rb') as f:
            test = pickle.load(f)
        return train['features'], train['labels'], valid['features'], valid['labels'], test['features'], test['labels']

    @classmethod
    def load_from_pickle_as_dataset(cls) -> List[TransformableDataset]:
        x_train, y_train, x_valid, y_valid, x_test, y_test = cls.load_from_pickle_tf()
        x_train = x_train.reshape((x_train.shape[0], x_train.shape[3], x_train.shape[1], x_train.shape[2]))
        x_valid = x_valid.reshape((x_valid.shape[0], x_valid.shape[3], x_valid.shape[1], x_valid.shape[2]))
        x_test = x_test.reshape((x_test.shape[0], x_test.shape[3], x_test.shape[1], x_test.shape[2]))
        train_dataset = TransformableDataset(
             torch.tensor(x_train, dtype=torch.float), torch.tensor(y_train, dtype=torch.long)
        )
        valid_dataset = TransformableDataset(
            torch.tensor(x_valid, dtype=torch.float), torch.tensor(y_valid, dtype=torch.long)
        )
        test_dataset = TransformableDataset(
            torch.tensor(x_test, dtype=torch.float), torch.tensor(y_test, dtype=torch.long)
        )
        return train_dataset, valid_dataset, test_dataset

    @classmethod
    def load_from_images(cls, dir_name: str) -> np.ndarray:
        dir_name = os.path.join(os.path.dirname(__file__), cls.DIR_NAME+'/'+dir_name)
        images = list()
        for img_file in sorted(os.listdir(dir_name)):
            images.append(np.array(Image.open(os.path.join(dir_name, img_file))))
        return np.array(images)
