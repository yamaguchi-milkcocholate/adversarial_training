from __future__ import annotations
import torch
import torch.optim as optim
import torch.nn as nn
from torchsummary import summary
from torch.utils.data.dataloader import DataLoader
import numpy as np
from time import time
from src.storage.modelrepo import ModelRepository
from src.storage.datasetrepo import GTSRBRepository
from src.trains.models import GTSRBCNN
from src.attacks.tools.pgd import multi_step_attack


class GTSRBAdversarialTraining:

    def __init__(self, batch_size: int, lr: float, wd: float):
        self.batch_size = batch_size
        self.lr = lr
        self.wd = wd
        self.input_size = (32, 32)
        self.input_range = (-0.5, 0.5)
        self.epochs = None

        self.train_loader, self.valid_loader, self.test_loader = self._prepare_dataset()

        if not torch.cuda.is_available():
            raise RuntimeError('GPU is not available.')
        self.device = torch.device("cuda:0")
        # device = torch.device("cpu")

        self.model = GTSRBCNN()
        self.model.to(self.device)
        summary(self.model, (3, 32, 32))

    def run(self, epochs: int, pdg_iteration: int):
        self.epochs = epochs
        start_at = time()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.wd)

        for epoch in range(self.epochs):
            running_loss = 0.0
            self.model.train()

            for i, (inputs, labels) in enumerate(self.train_loader, 0):
                optimizer.zero_grad()
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                delta = multi_step_attack(
                    model=self.model, X=inputs, y=labels, input_range=self.input_range,
                    epsilon=8 / 255, alpha=4 / 255, num_iter=pdg_iteration, randomize=True
                )
                inputs += delta
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
            self._evaluate(epoch=epoch + 1, running_loss=running_loss)

        self.model.eval()
        print('Accuracy: {:.2f} %%'.format(self._accuracy(data_loader=self.test_loader)))
        ModelRepository.save(filename='GTSRB/pdg_model', model=self.model)
        print(time() - start_at)

    def _prepare_dataset(self):
        train_set, valid_set, test_set = GTSRBRepository.load_from_pickle_as_dataset()
        print('Train Data Size:', str(len(train_set.data)))
        print('Valid Data Size: ', str(len(valid_set)))
        print('Test Data Size:', str(len(test_set.data)))
        classes = tuple(
            np.linspace(0, len(np.unique(train_set.target)) - 1, len(np.unique(train_set.target)), dtype=np.uint8))
        print('Class:', *classes)

        train_loader = DataLoader(
            train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2
        )
        valid_loader = DataLoader(
            valid_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2
        )
        test_loader = DataLoader(
            test_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2
        )
        return train_loader, valid_loader, test_loader

    def _evaluate(self, epoch: int, running_loss: float):
        self.model.eval()
        print('[{:d}/{:d}] loss: {:.3f} train acc: {:.3f} valid acc: {:.3f} test acc: {:.3f}'.format(
            epoch,
            self.epochs,
            running_loss / self.batch_size,
            self._accuracy(data_loader=self.train_loader, is_for_all=False),
            self._accuracy(data_loader=self.valid_loader),
            self._accuracy(data_loader=self.test_loader)
        ))

    def _accuracy(self, data_loader, is_for_all: bool = True) -> float:
        correct = 0
        total = 0
        with torch.no_grad():
            for (images, labels) in data_loader:
                images = images.to(self.device, dtype=torch.float)
                labels = labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                if not is_for_all:
                    break
        return 100 * float(correct / total)
