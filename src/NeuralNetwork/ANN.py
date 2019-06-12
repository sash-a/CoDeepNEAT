import torch.nn as nn
import torch.nn.functional as F

class ModuleNet(nn.Module):
    def __init__(self, moduleGraph):
        super(ModuleNet, self).__init__()
        self.moduleGraph = moduleGraph

    def forward(self, x):
        x = self.moduleGraph.passANNInputUpGraph(x)
        return x

