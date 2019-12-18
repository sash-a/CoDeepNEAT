import torch
import torch.nn.functional as F
from data import DataManager
from torch import nn, optim

from src.Config import Config
from src.Utilities import Utils

from src2.Phenotype.NeuralNetwork.Aggregation.Merger import merge


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.input = nn.Conv2d(3,41,kernel_size=7, padding=3)  # 28
        self.c1 = nn.Conv2d(41,42,kernel_size=7, padding=3)  # 22/2=11
        self.c2 = nn.Conv2d(42,41,kernel_size=7,padding=3)  # 5/2
        self.c3 = nn.Conv2d(41,42,kernel_size=7)  # 10

        self.final = nn.Linear(168,10)

        self.b1 = nn.BatchNorm2d(42)
        self.b3 = nn.BatchNorm2d(42)

        self.m = nn.MaxPool2d(2)

        self.optimizer: optim.adam = optim.Adam(self.parameters(), lr=0.001,
                                                betas=(0.9, 0.999))
        self.loss_fn = nn.NLLLoss()  # TODO mutagen

    # Defining the forward pass
    def forward(self, x):

        x = self.input(x)
        x = self.m(x)

        x = self.c1(x)
        x = self.b1(x)

        x = self.c2(x)
        x = self.m(x)

        x = self.c3(x)
        x = self.b3(x)
        # print(x.size())

        batch_size = list(x.size())[0]
        x = self.final(x.view(batch_size,-1))

        return x
