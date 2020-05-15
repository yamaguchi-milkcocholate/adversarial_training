import torch
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


class GTSRBCNN(nn.Module):

    def __init__(self):
        super(GTSRBCNN, self).__init__()
        self.conv0 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=1, padding=0)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout2d()

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout2 = nn.Dropout2d()

        self.conv5 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=2)
        self.conv6 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=5, stride=1, padding=2)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout3 = nn.Dropout2d()

        self.fc1 = nn.Linear(32 * 16 * 16 + 64 * 8 * 8 + 128 * 4 * 4, 1024)
        self.dropout4 = nn.Dropout2d()
        self.fc2 = nn.Linear(1024, 1024)
        self.dropout5 = nn.Dropout2d()
        self.fc3 = nn.Linear(1024, 43)
        self._init_weight()

    def forward(self, x):
        x = F.relu(self.conv0(x))
        x = F.relu(self.conv1(x))
        x = self.pool1(F.relu(self.conv2(x)))
        x = self.dropout1(x)
        flatten1 = x.clone().view(-1, 32 * 16 * 16)

        x = F.relu(self.conv3(x))
        x = self.pool2(F.relu(self.conv4(x)))
        x = self.dropout2(x)
        flatten2 = x.clone().view(-1, 64 * 8 * 8)

        x = F.relu(self.conv5(x))
        x = self.pool3(F.relu(self.conv6(x)))
        x = self.dropout3(x)
        x = x.view(-1, 128 * 4 * 4)
        x = torch.cat((flatten1, flatten2, x), 1)
        x = self.fc1(x)
        x = self.dropout4(x)
        x = self.fc2(x)
        x = self.dropout5(x)
        x = self.fc3(x)
        return x

    def _init_weight(self):
        torch.nn.init.kaiming_normal_(self.conv0.weight)
        torch.nn.init.kaiming_normal_(self.conv1.weight)
        torch.nn.init.kaiming_normal_(self.conv2.weight)
        torch.nn.init.kaiming_normal_(self.conv3.weight)
        torch.nn.init.kaiming_normal_(self.conv4.weight)
        torch.nn.init.kaiming_normal_(self.conv5.weight)
        torch.nn.init.kaiming_normal_(self.conv6.weight)
        torch.nn.init.kaiming_normal_(self.fc1.weight)
        torch.nn.init.kaiming_normal_(self.fc2.weight)
        torch.nn.init.kaiming_normal_(self.fc3.weight)


