from torch import nn
from torch import optim
import torch


class ModuleNet(nn.Module):
    def __init__(self, moduleGraph, lr=0.001, beta1=0.9, beta2=0.999, loss_fn=nn.MSELoss(), outputDimensionality = torch.tensor([1])):
        super(ModuleNet, self).__init__()
        self.moduleGraph = moduleGraph
        self.loss_fn = loss_fn
        self.lr = lr
        self.outputDimensionality = outputDimensionality
        outputNodes = torch.cumprod( outputDimensionality,0)[0].item()
        print(moduleGraph.getOutputNode().getDimensionality() ,outputNodes )
        self.finalLayer = nn.Linear(moduleGraph.getOutputNode().getDimensionality(), outputNodes)

        self.optimizer = optim.Adam(moduleGraph.getParameters({}), lr=self.lr, betas=(beta1, beta2))

    def forward(self, x):
        x = self.moduleGraph.passANNInputUpGraph(x)
        if(not self.outputDimensionality is None):
            batchSize = x.size()[0]
            #print("output shape:",x.size(), "flat:",x.view(batchSize,-1).size())
            x = self.finalLayer(x.view(batchSize,-1))
            #print(self.outputDimensionality[0].item())
            #only works with 1 output dimension
            x = x.view(batchSize, self.outputDimensionality[0].item(), -1)

        #print("exiting forward pass with output:", x.squeeze().size())
        return x.squeeze()
