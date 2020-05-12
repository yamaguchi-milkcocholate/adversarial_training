import pickle
import os


class ModelRepository:

    @classmethod
    def save(cls, filename, model):
        path = os.path.join(os.path.dirname(__file__)+'/models', filename)

        if os.path.exists(path):
            os.remove(path)
        with open(os.path.join(os.path.dirname(__file__)+'/models', filename), 'wb') as f:
            pickle.dump(model, f)
