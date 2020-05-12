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


def train(batch_size: int, epochs: int):
    start_at = time()
    train_set, test_set = GTSRBRepository.load_from_pickle()
    print('Train Data Size:', str(len(train_set.data)))
    print('Test Data Size:', str(len(test_set.data)))
    classes = tuple(np.linspace(0, len(np.unique(train_set.target))-1, len(np.unique(train_set.target)), dtype=np.uint8))
    print('Class:', *classes)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )

    if not torch.cuda.is_available():
        raise RuntimeError('GPU is not available.')
    device = torch.device("cuda:0")
    net = GTSRBCNN()
    net.to(device)
    summary(net, (3, 32, 32))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader, 0):
            inputs.to(device)
            labels.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
        print('[{:d},] loss: {:.3f}'.format(epoch + 1, running_loss))
    print('Finished Training')

    correct = 0
    total = 0
    with torch.no_grad():
        for (images, labels) in test_loader:
            images.to(device)
            labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy: {:.2f} %%'.format(100 * float(correct / total)))
    ModelRepository.save(filename='GTSRB/model.p', model=net)
    print(time() - start_at)
