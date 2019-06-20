from torch import nn
from torch import optim
import torch
import torch.nn.functional as F


class ModuleNet(nn.Module):
    def __init__(self, moduleGraph, lr=0.001, beta1=0.9, beta2=0.999, loss_fn=nn.MSELoss()):
        super(ModuleNet, self).__init__()
        self.moduleGraph = moduleGraph
        self.loss_fn = loss_fn
        self.lr = lr
        self.dimensionalityConfigured = False
        self.outputDimensionality = None
        self.optimizer = optim.Adam(moduleGraph.getParameters({}), lr=self.lr, betas=(beta1, beta2))

    def specifyOutputDimensionality(self,inputSample, outputDimensionality = torch.tensor([1]), device =  torch.device("cpu")):
        if(self.dimensionalityConfigured):
            print("warning - trying to configure dimensionality multiple times on the same network")
            return
        print("configuring output dims with in=",inputSample.size(),end = " ")
        outputNodes = self.getFlatNumber(outputDimensionality,0)
        output = self(inputSample)
        inLayers = self.getFlatNumber(output)
        print("out = ", output.size(), "using linear layer (",inLayers,",",outputNodes,")")

        self.finalLayer = nn.Linear(inLayers, 500).to(device)
        self.final2 = nn.Linear(500,outputNodes).to(device)
        self.dimensionalityConfigured = True
        self.outputDimensionality = outputDimensionality

    def getFlatNumber(self, tensor, startDim = 1):
        prod = 1
        items = list(tensor.size())
        for i in range(startDim,len(items)):
            prod*=items[i]

        return prod

    def forward(self, x):
        x = self.moduleGraph.passANNInputUpGraph(x)
        if(self.dimensionalityConfigured):
            batchSize = x.size()[0]
            x = F.relu(self.finalLayer(x.view(batchSize,-1)))

            x = self.final2(x)
            #only works with 1 output dimension
            x = x.view(batchSize, self.outputDimensionality[0].item(), -1)

        #print("exiting forward pass with output:", x.squeeze().size())
        return x.squeeze()
