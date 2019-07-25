import torch.nn as nn
import torch.nn.functional as F


class SmallNet(nn.Module):
    def __init__(self):
        super(SmallNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class MediumNet(nn.Module):

    def __init__(self):
        super(MediumNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 12, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(12, 32, 5)
        self.fc1 = nn.Linear(32 * 5 * 5, 240)
        self.fc2 = nn.Linear(240, 160)
        self.fc3 = nn.Linear(160, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class LargeNet(nn.Module):

    def __init__(self):
        super(LargeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 24, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(24, 64, 5)
        self.fc1 = nn.Linear(64 * 5 * 5, 480)
        self.fc2 = nn.Linear(480, 320)
        self.fc3 = nn.Linear(320, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class LargeBatchNormNet(LargeNet):

    def __init__(self):
        super(LargeBatchNormNet, self).__init__()
        self.batch_norm_1d_1 = nn.BatchNorm1d(480)
        self.batch_norm_1d_2 = nn.BatchNorm1d(320)

        self.batch_norm_2d_1 = nn.BatchNorm2d(24)
        self.batch_norm_2d_2 = nn.BatchNorm2d(64)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.batch_norm_2d_1(x)
        x = self.pool(F.relu(self.conv2(x)))
        x = self.batch_norm_2d_2(x)
        x = x.view(-1, 64 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = self.batch_norm_1d_1(x)
        x = F.relu(self.fc2(x))
        x = self.batch_norm_1d_2(x)
        x = self.fc3(x)
        return x






class DropOutNet(SmallNet):
    """
    performs worse than standard met despite changing where and how much drop out
    """

    def __init__(self, drop_out_factor = 0.03):
        super(DropOutNet,self).__init__()
        self.drop_out_2d = nn.Dropout2d(drop_out_factor/3)
        self.drop_out_1d = nn.Dropout(drop_out_factor)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.drop_out_2d(x)
        x = self.pool(F.relu(self.conv2(x)))
        x = self.drop_out_2d(x)
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = self.drop_out_1d(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


class BatchNormNet(SmallNet):

    def __init__(self):
        super(BatchNormNet,self).__init__()
        self.batch_norm_1d_1 = nn.BatchNorm1d(120)
        self.batch_norm_1d_2 = nn.BatchNorm1d(84)

        self.batch_norm_2d_1 = nn.BatchNorm2d(6)
        self.batch_norm_2d_2 = nn.BatchNorm2d(16)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.batch_norm_2d_1(x)
        x = self.pool(F.relu(self.conv2(x)))
        x = self.batch_norm_2d_2(x)
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = self.batch_norm_1d_1(x)
        x = F.relu(self.fc2(x))
        x = self.batch_norm_1d_2(x)
        x = self.fc3(x)

        return x


