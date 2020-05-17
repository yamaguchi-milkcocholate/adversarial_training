import pickle
import os
from torch.nn import Module
import torch


class ModelRepository:

    @classmethod
    def save(cls, filename, model):
        torch.save(model, os.path.dirname(__file__)+'/models', filename)

    @classmethod
    def load(cls, filename) -> Module:
        with open(os.path.join(os.path.dirname(__file__)+'/models', filename), 'rb') as f:
            pickle.load(f)
