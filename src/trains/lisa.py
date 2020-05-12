from __future__ import annotations
import torch
import torchvision
import torch.optim as optim
import torchvision.transforms as transforms
from torchsummary import summary
import torch
import numpy as np
from time import time
from src.storage.datasetrepo import LISARepository
from src.trains.models import LISACNN


def pre_process(used_for: str):
    if used_for == 'train':
        return LISARepository.load(
            used_for='train',
            transformer=transforms.Compose([
                transforms.Resize(size=(32, 32)),
                # transforms.RandomResizedCrop(32),
                # transforms.RandomHorizontalFlip(),
                # transforms.ColorJitter(hue=.1, saturation=.1),
                transforms.ToTensor(),
            ]))
    elif used_for == 'valid':
        LISARepository.load(
            used_for='valid',
            transformer=transforms.Compose([
                transforms.Resize(size=(32, 32)),
                transforms.ToTensor(),
            ]))
    elif used_for == 'test':
        LISARepository.load(
            used_for='test',
            transformer=transforms.Compose([
                transforms.Resize(size=(32, 32)),
                transforms.ToTensor(),
            ]))


def init_weights(model):
    if isinstance(model, torch.nn.Conv2d):
        torch.nn.init.xavier_uniform_(model.weight)
        torch.nn.init.zeros_(model.bias)


def train(batch_size: int, epochs: int):
    data_loaders = {
        use: torch.utils.data.DataLoader(
                pre_process(used_for=use), batch_size=batch_size, shuffle=True
            ) for use in ['train', 'val', 'test']
    }
    classes = tuple(pre_process(used_for='test').classes)

    print(*classes)
    print('train | valid | test')
    print('{0} | {1} | {2}'.format(*[str(len(data_loaders[i])) for i in data_loaders]))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = LISACNN()
    model.to(device)
    print(device)
    summary(model, (3, 32, 32))
    model.apply(init_weights)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.03, momentum=0.9)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)

    for epoch_i in epochs:
        print('Epoch {0}/{1}'.format(epoch_i, epochs))


if __name__ == '__main__':
    start_at = time()
    train(batch_size=64, epochs=2)
    print(time() - start_at)
