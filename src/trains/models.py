import torch.nn as nn
import torch.nn.functional as F


class LISACNN(nn.Module):

    def __init__(self):
        super(LISACNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 8, stride=2, padding=3)
        self.conv2 = nn.Conv2d(64, 128, 6, stride=2, padding=0)
        self.conv3 = nn.Conv2d(128, 128, 5, stride=1, padding=0)
        self.fc1 = nn.Linear(512, 16)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x
