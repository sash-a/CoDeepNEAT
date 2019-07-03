import torch
from torch import nn, optim
import torch.nn.functional as F
from src.Utilities import Utils


class ModuleNet(nn.Module):
    def __init__(self, module_graph, lr=0.001, beta1=0.9, beta2=0.999, loss_fn=nn.MSELoss()):
        super(ModuleNet, self).__init__()
        self.module_graph = module_graph
        self.loss_fn = loss_fn
        self.lr = lr
        self.dimensionality_configured = False
        self.outputDimensionality = None
        # self.optimizer = optim.Adam(module_graph.get_parameters({}), lr=self.lr, betas=(beta1, beta2))
        self.beta1 = beta1
        self.beta2 = beta2
        self.optimizer = None

    def specify_dimensionality(self, input_sample, output_dimensionality=torch.tensor([1]),
                               device=torch.device("cpu")):
        if self.dimensionality_configured:
            print("warning - trying to configure dimensionality multiple times on the same network")
            return
        #print("configuring output dims with in=", input_sample.size())

        #self.module_graph.add_reshape_node(list(input_sample.size()))

        output_nodes = Utils.get_flat_number(output_dimensionality, 0)
        output = self(input_sample, configuration_run = True)
        if(output is None):
            raise Exception("Error: failed to pass input through nn")
        in_layers = Utils.get_flat_number(output)
        # print("out = ", output.size(), "using linear layer (", in_layers, ",", output_nodes, ")")

        self.final_layer = nn.Linear(in_layers, output_nodes).to(device)
        self.dimensionality_configured = True
        self.outputDimensionality = output_dimensionality

        self.optimizer = optim.Adam(self.module_graph.get_parameters({}), lr=self.lr, betas=(self.beta1, self.beta2))

    def forward(self, x, configuration_run = False):
        if (x is None):
            print("null x passed to forward 1")
            return
        x = self.module_graph.pass_ann_input_up_graph(x, configuration_run = configuration_run)
        if (x is None):
            print("received null output from module graph given non null input")
            return

        if self.dimensionality_configured:

            batch_size = x.size()[0]
            x = F.relu(self.final_layer(x.view(batch_size, -1)))
            # only works with 1 output dimension
            x = x.view(batch_size, self.outputDimensionality[0].item(), -1)
        else:
            # print("dimensionality of a net not configured x==none~", (x is None))
            pass

        return x.squeeze()
