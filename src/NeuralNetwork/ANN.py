from torch import nn
from torch import optim


class ModuleNet(nn.Module):
    def __init__(self, moduleGraph, lr=0.001, beta1=0.9, beta2=0.999, loss_fn=nn.MSELoss()):
        super(ModuleNet, self).__init__()
        self.moduleGraph = moduleGraph
        self.loss_fn = loss_fn
        self.lr = lr
        # TODO self.parameters()
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr, betas=(beta1, beta2))

    def forward(self, x):
        x = self.moduleGraph.passANNInputUpGraph(x)
        print("exiting forward pass with output:", x)
        return x
