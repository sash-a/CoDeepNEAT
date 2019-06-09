"""Custom layers that need to be added to an instance of Net"""

from torch import nn


class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, input):
        return input.view(self.shape)
