import torch
import torch.nn.functional as F
from data import DataManager
from torch import nn, optim

from src.Config import Config
from src.Utilities import Utils


class ModuleNet(nn.Module):

    def __init__(self, module_graph, loss_fn=F.nll_loss):
        super(ModuleNet, self).__init__()
        self.module_graph = module_graph
        self.loss_fn = loss_fn
        self.lr = 0
        self.effective_lr = 0
        self.beta1 = -1
        self.beta2 = -1
        self.dimensionality_configured = False
        self.outputDimensionality = None
        self.optimizer = None

        self.final_layer = None

    def configure(self, learning_rate, beta1, beta2):
        """sets the hyper parameters which are evolved by the blueprint"""
        self.lr = learning_rate
        self.effective_lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2

    def specify_dimensionality(self, input_sample, output_dimensionality=torch.tensor([10])):
        """configures the final layers dimensionality to fit the target functions dimensions.
            performs a 'configuration run' which passes a sample input up the layer graph once.
            this is used to reshape inputs as they go through the graph.
        """
        if self.dimensionality_configured:
            print("Warning - trying to configure dimensionality multiple times on the same network")
            return
        if self.lr == 0:
            raise Exception('Must set net learning rate before calling specify dimensionality')
        output_nodes = int(list(output_dimensionality)[0])
        output = self(input_sample, configuration_run=True)
        if output is None:
            raise Exception("Error: failed to pass input through nn")

        in_layers = Utils.get_flat_number(output)

        self.final_layer = nn.Linear(in_layers, output_nodes).to(Config.get_device())

        self.dimensionality_configured = True
        self.outputDimensionality = output_dimensionality
        final_params = self.final_layer.parameters()
        full_parameters = self.module_graph.module_graph_root_node.get_parameters({})
        full_parameters.extend(final_params)

        self.optimizer = optim.Adam(full_parameters, lr=self.lr, betas=(self.beta1, self.beta2))

        self.init_weights()
        self.module_graph.blueprint_genome.weight_init.get_value()(self.final_layer.weight)

    def forward(self, x, configuration_run=False):
        """passes an input up and through the layer graph.
            then passes it through the final layer, which
            shapes the output to the target functions dimensionality
        """
        if x is None:
            print("null x passed to forward 1")
            return
        x = self.module_graph.module_graph_root_node.pass_ann_input_up_graph(x, configuration_run=configuration_run)
        if x is None:
            print("received null output from module graph given non null input")
            return

        if self.dimensionality_configured:
            batch_size = x.size()[0]
            x = F.relu(self.final_layer(x.view(batch_size, -1)))
            # only works with 1 output dimension
            x = x.view(batch_size, self.outputDimensionality[0].item(), -1)

        return torch.squeeze(F.log_softmax(x, dim=1))

    def init_weights(self):
        self._init_weights(self.module_graph.module_graph_root_node)

    def _init_weights(self, module_node):
        """uses the weight init scheme from the blueprint to initialise
        the weights of all the layers in each module node"""
        for child in module_node.children:
            if child.deep_layer is None:
                continue

            self.module_graph.blueprint_genome.weight_init.get_value()(child.deep_layer.weight)
            self._init_weights(child)

    def multiply_learning_rate(self, factor):
        """used for adaptive learning rate adjustment"""
        new_lr = self.lr * factor

        for param_group in self.optimizer.param_groups:
            if new_lr != param_group['lr'] or Config.use_adaptive_learning_rate_adjustment:
                updated_lr = "updating lr from " + repr(param_group['lr']) + " to " + (repr(param_group['lr']*factor) if Config.use_adaptive_learning_rate_adjustment else repr(self.lr * factor))
                print(updated_lr)
                with open(DataManager.get_results_file(), 'a+') as f:
                    f.write(updated_lr)
                    f.write('\n')

                if Config.use_adaptive_learning_rate_adjustment:
                    param_group['lr'] *= factor
                else:
                    param_group['lr'] = new_lr


def create_nn(module_graph, sample_inputs, feature_multiplier=1):
    """static method to create an instance of a module net given a module graph."""
    blueprint_individual = module_graph.blueprint_genome

    if module_graph is None:
        raise Exception("None module graph produced from blueprint")
    try:
        net = module_graph.to_nn(
            in_features=module_graph.module_graph_root_node.get_first_feature_count(sample_inputs)).to(
            Config.get_device())

    except Exception as e:
        raise Exception("Error: failed to parse module graph into nn", e)

    for module_node in module_graph.module_graph_root_node.get_all_nodes_via_bottom_up(set()):
        module_node.generate_module_node_from_gene(feature_multiplier=feature_multiplier)

    net.configure(blueprint_individual.learning_rate(), blueprint_individual.beta1(), blueprint_individual.beta2())
    net.specify_dimensionality(sample_inputs)

    return net
