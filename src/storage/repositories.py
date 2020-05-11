from torchvision.transforms import Compose
from torchvision.datasets import ImageFolder
import os
from abc import ABC


class BaseRepository(ABC):
    DIR_NAME: str

    @classmethod
    def load(cls, used_for: str, transformer: Compose):
        if used_for in ['train', 'valid', 'test']:
            return ImageFolder(os.path.join(cls.DIR_NAME, used_for), transformer)
        else:
            raise FileNotFoundError()


class LISARepository(BaseRepository):
    DIR_NAME = 'LISA'

    @classmethod
    def load(cls, used_for: str, transformer: Compose):
        return super().load(used_for=used_for, transformer=transformer)


class GTSRBRepository(BaseRepository):
    DIR_NAME = 'GTSRB'

    @classmethod
    def load(cls, used_for: str, transformer: Compose):
        return super().load(used_for=used_for, transformer=transformer)
