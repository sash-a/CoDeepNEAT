import torch
from torch import nn, optim
import torch.nn.functional as F
from src.Utilities import Utils


class ModuleNet(nn.Module):
    def __init__(self, module_graph, lr=0.001, beta1=0.9, beta2=0.999, loss_fn=nn.MSELoss()):
        super(ModuleNet, self).__init__()
        self.moduleGraph = module_graph
        self.loss_fn = loss_fn
        self.lr = lr
        self.dimensionality_configured = False
        self.outputDimensionality = None
        # self.optimizer = optim.Adam(module_graph.get_parameters({}), lr=self.lr, betas=(beta1, beta2))
        params = module_graph.get_parameters({})
        if (params is None):
            print("null parameters object gotten from module graph", module_graph)
        self.optimizer = optim.Adam(params, lr=self.lr, betas=(beta1, beta2))

    def specify_output_dimensionality(self, input_sample, output_dimensionality=torch.tensor([1]),
                                      device=torch.device("cpu")):
        if self.dimensionality_configured:
            print("warning - trying to configure dimensionality multiple times on the same network")
            return
        # print("configuring output dims with in=", input_sample.size(), end=" ")
        output_nodes = Utils.get_tensor_flat_number(output_dimensionality, 0)
        output = self(input_sample)
        in_layers = Utils.get_tensor_flat_number(output)
        # print("out = ", output.size(), "using linear layer (", in_layers, ",", output_nodes, ")")

        self.final_layer = nn.Linear(in_layers, output_nodes).to(device)
        self.dimensionality_configured = True
        self.outputDimensionality = output_dimensionality

    def forward(self, x):
        if (x is None):
            print("null x passed to forward 1")
            return
        x = self.moduleGraph.pass_ann_input_up_graph(x)
        if (x is None):
            print("received null output from module graph given non null input")
            return

        if self.dimensionality_configured:

            batch_size = x.size()[0]
            x = x.view(batch_size, -1)
            if (x is None):
                print("null x 2")
            x = self.final_layer(x)
            if (x is None):
                print("null x 2")
            x = F.relu(x)
            if (x is None):
                print("null x 2")
            # x = F.relu(self.final_layer(x.view(batch_size, -1)))
            # only works with 1 output dimension
            x = x.view(batch_size, self.outputDimensionality[0].item(), -1)
        else:
            # print("dimensionality of a net not configured x==none~", (x is None))
            pass

        return x.squeeze()
