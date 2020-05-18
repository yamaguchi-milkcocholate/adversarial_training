import pickle
import os
from torch.nn import Module
import torch


class ModelRepository:

    @classmethod
    def save(cls, filename, model):
        torch.save(model, os.path.join(os.path.dirname(__file__)+'/models', filename))

    @classmethod
    def load(cls, filename, model: Module, device) -> Module:
        model.load_state_dict(torch.load(
            os.path.join(os.path.dirname(__file__)+'/models', filename),
            map_location=device)
        )
        return model
