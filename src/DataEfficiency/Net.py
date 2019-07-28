import torch.nn as nn
import torch.nn.functional as F
from data import DataManager
import os

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

    def get_name(self):
        return repr(type(self)).split(".")[-1].split("'")[0] + "_" + repr(self.size)+".txt"

    def get_results_file_name(self, verbose):
        if verbose:
            return os.path.join(DataManager.get_DataEfficiencyResults_folder(), "Verbose" , self.get_name())
        else:
            return os.path.join(DataManager.get_DataEfficiencyResults_folder(), "Summarised" , self.get_name())

    def save_results(self,set_size_accuracy_tuples, verbose):
        f = open(self.get_results_file_name(verbose), "w+")
        print("saving results:",set_size_accuracy_tuples,"to",self.get_results_file_name(verbose))

        for tuple in set_size_accuracy_tuples:
            set_size, accuracy = tuple
            writable = repr(round(set_size,2))+":"+repr(accuracy)
            f.write(writable + "\n")
        f.close()

    def does_net_have_results_file(self, verbose):
        return os.path.exists(self.get_results_file_name(verbose))


        


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

