import torch.nn as nn

class ModuleNet(nn.Module):

    def __init__(self, moduleGraph):
        super(ModuleNet, self).__init__()
        self.moduleGraph = moduleGraph

    def forward(self, x):
        x = self.moduleGraph.passANNInputUpGraph(x)
        print("exiting forward pass with output:",x)
        return x

