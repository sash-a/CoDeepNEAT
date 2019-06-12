"""Custom layers that need to be added to an instance of Net"""

from torch import nn, cat
import torch


class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, input):
        return input.view(self.shape)


# This is a base class should never be called direct because it doesn't have a forward
class Merge(nn.Module):
    def __init__(self, children, parent):
        super(Merge, self).__init__()
        # if len(children < 2):
        #     raise Exception('Tried to merge less than two objects')

        self.children = children
        self.parent = parent

    def forward(self, input):
        raise Exception('Use a specific type of merge not the base class')


class MergeSum(Merge):
    def __init__(self, children, parent):
        super(MergeSum, self).__init__(children, parent)

    def forward(self, input):
        res = [y(input) for y in self.children]
        joined = torch.sum(torch.stack(res).cuda(), dim=0)  #

        return self.parent(joined)


class MergeCat(Merge):
    def __init__(self, children, parent):
        super(MergeCat, self).__init__(children, parent)

    def forward(self, input):
        res = []
        print(self.children)
        for y in self.children:
            res.append(y(input))
        # res = [y(input) for y in self.children]
        joined = cat(res, dim=0)

        return self.parent(joined)
