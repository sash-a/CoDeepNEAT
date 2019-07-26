import torch.nn as nn
import torch.nn.functional as F
from data import DataManager

class StandardNet(nn.Module):

    def __init__(self, size):
        super(StandardNet, self).__init__()
        self.size = size

        self.conv1 = nn.Conv2d(3, 6*size, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6*size, 16*size, 5)
        self.fc1 = nn.Linear(16 * 5 * 5 * size, 120*size)
        self.fc2 = nn.Linear(120*size, 84*size)
        self.fc3 = nn.Linear(84*size, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5 * self.size)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def _get_name(self):
        return type(self)

    def save_results(self,set_size_accuracy_tuples):
        f = open(DataManager.get_DataEfficiencyResults_folder())
        


class DropOutNet(StandardNet):
    """
    performs worse than standard met despite changing where and how much drop out
    """

    def __init__(self, size, drop_out_factor = 0.03):
        super(DropOutNet, self).__init__(size)
        self.drop_out_2d = nn.Dropout2d(drop_out_factor/4)#literature shows convs suffer from high drop out
        self.drop_out_1d = nn.Dropout(drop_out_factor)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.drop_out_2d(x)
        x = self.pool(F.relu(self.conv2(x)))
        x = self.drop_out_2d(x)
        x = x.view(-1, 16 * 5 * 5 * self.size)
        x = F.relu(self.fc1(x))
        x = self.drop_out_1d(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

class BatchNormNet(StandardNet):

    def __init__(self, size):
        super(BatchNormNet, self).__init__(size)
        self.batch_norm_1d_1 = nn.BatchNorm1d(120*size)
        self.batch_norm_1d_2 = nn.BatchNorm1d(84*size)

        self.batch_norm_2d_1 = nn.BatchNorm2d(6*size)
        self.batch_norm_2d_2 = nn.BatchNorm2d(16*size)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.batch_norm_2d_1(x)
        x = self.pool(F.relu(self.conv2(x)))
        x = self.batch_norm_2d_2(x)
        x = x.view(-1, 16 * 5 * 5 * self.size)
        x = F.relu(self.fc1(x))
        x = self.batch_norm_1d_1(x)
        x = F.relu(self.fc2(x))
        x = self.batch_norm_1d_2(x)
        x = self.fc3(x)

        return x


class ExtraSmallNet(nn.Module):
    def __init__(self):
        super(ExtraSmallNet, self).__init__()
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

class SmallNet(nn.Module):

    def __init__(self):
        super(SmallNet, self).__init__()
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


class mediumNet(nn.Module):

    def __init__(self):
        super(mediumNet, self).__init__()
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

class LargeNet(nn.Module):

    def __init__(self):
        super(LargeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 48, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(48, 128, 5)
        self.fc1 = nn.Linear(128 * 5 * 5, 1000)
        self.fc2 = nn.Linear(1000, 640)
        self.fc3 = nn.Linear(640, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 128 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class mediumBatchNormNet(mediumNet):

    def __init__(self):
        super(mediumBatchNormNet, self).__init__()
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






class SmallDropOutNet(ExtraSmallNet):
    """
    performs worse than standard met despite changing where and how much drop out
    """

    def __init__(self, drop_out_factor = 0.03):
        super(SmallDropOutNet, self).__init__()
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


class ExtraSmallBatchNormNet(ExtraSmallNet):

    def __init__(self):
        super(ExtraSmallBatchNormNet, self).__init__()
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


