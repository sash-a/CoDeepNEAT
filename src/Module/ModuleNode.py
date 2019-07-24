from src.Graph.Node import Node
from src.Module.ReshapeNode import ReshapeNode
from src.Utilities import Utils
import torch.nn as nn
import torch.nn.functional as F
import math
from src.NeuralNetwork.Net import ModuleNet
from src.Config import Config

minimum_conv_dim = 8


class ModuleNode(Node):
    """
    ModuleNode represents a node in a module
    The whole  Module is represented as the input Module, followed by its children
    Each Module should have one input node and one output node
    All module children are closer to the output node than their parent (one way/ feed forward)

    Modules get parsed into neural networks via traversals
    """

    layerType = None
    reduction = None
    regularisation = None

    def __init__(self, module_NEAT_node, module_genome):
        Node.__init__(self)
        self.traversed = False

        self.deep_layer = None  # an nn layer object such as    nn.Conv2d(3, 6, 5) or nn.Linear(84, 10)
        self.in_features = -1
        self.out_features = -1
        self.activation = None

        self.reduction = None
        self.regularisation = None
        self.dropout = None

        self.reshape = None  # reshapes the input given before passing it through this nodes deeplayer

        self.module_NEAT_genome = module_genome
        self.module_NEAT_node = module_NEAT_node

        self.fitness_values = []

        if not (module_NEAT_node is None):
            self.generate_module_node_from_gene()

    def generate_module_node_from_gene(self):
        # print("generating module node from gene:", self.module_NEAT_node)
        self.out_features = self.module_NEAT_node.layer_type.get_sub_value("out_features")
        self.activation = self.module_NEAT_node.activation()

        neat_regularisation = self.module_NEAT_node.layer_type.get_sub_value("regularisation", return_mutagen=True)
        neat_reduction = self.module_NEAT_node.layer_type.get_sub_value("reduction", return_mutagen=True)
        neat_dropout = self.module_NEAT_node.layer_type.get_sub_value("dropout", return_mutagen=True)

        if not (neat_regularisation() is None):
            self.regularisation = neat_regularisation()(self.out_features)

        if not (neat_reduction is None) and not (neat_reduction() is None):
            if neat_reduction() == nn.MaxPool2d or neat_reduction() == nn.MaxPool1d:
                pool_size = neat_reduction.get_sub_value("pool_size")
                if neat_reduction() == nn.MaxPool2d:
                    self.reduction = nn.MaxPool2d(pool_size, pool_size)
                if neat_reduction() == nn.MaxPool1d:
                    self.reduction = nn.MaxPool1d(pool_size)
            else:
                print("Error not implemented reduction ", neat_reduction())

        if not (neat_dropout is None) and not (neat_dropout() is None):
            self.dropout = neat_dropout()(neat_dropout.get_sub_value("dropout_factor"))

    def to_nn(self, in_features, print_graphs=False):
        self.create_layer(in_features)
        if print_graphs:
            self.plot_tree_with_matplotlib()
        return ModuleNet(self)

    def create_layer(self, in_features):

        self.in_features = in_features
        device = Config.get_device()

        layer_type = self.module_NEAT_node.layer_type
        if layer_type() == nn.Conv2d:
            try:
                self.deep_layer = nn.Conv2d(self.in_features, self.out_features,
                                            kernel_size=layer_type.get_sub_value("conv_window_size"),
                                            stride=layer_type.get_sub_value("conv_stride"))
                try:
                    self.deep_layer = self.deep_layer.to(device)
                except Exception as e:
                    print(e)
                    raise Exception("created conv layer - but failed to move it to device" + repr(device))

            except Exception as e:
                print("Error:", e)
                print("failed to create conv", self, self.in_features, self.out_features,
                      layer_type.get_sub_value("conv_window_size"),
                      layer_type.get_sub_value("conv_stride"), "deep layer:", self.deep_layer)
                print('Module with error', self.module_NEAT_genome.connections)

        else:
            self.deep_layer = layer_type()(self.in_features, self.out_features).to(device)

        if not (self.reduction is None):
            self.reduction = self.reduction.to(device)

        if not (self.regularisation is None):
            self.regularisation = self.regularisation.to(device)

    def delete_layer(self):
        self.deep_layer = None
        self.reduction = None
        self.regularisation = None
        self.dropout = None

    def delete_all_layers(self):
        if not self.is_input_node():
            raise Exception("must be called on root node")

        for node in self.get_all_nodes_via_bottom_up(set()):
            node.delete_layer()

    # could be made more efficient as a breadth first instead of depth first because of duplicate paths
    def insert_aggregator_nodes(self, state="start"):
        from src.Module.AggregatorNode import AggregatorNode as Aggregator
        """
        *NB  -- must have called getTraversalIDs on root node to use this function
        traverses the module graph from the input up to output, and inserts aggregators
        which are module nodes which take multiple inputs and combine them"""

        if state == "start":  # method is called from the root node - but must be traversed from the output node
            output_node = self.get_output_node()
            output_node.insert_aggregator_nodes("fromTop")
            return

        num_parents = len(self.parents)

        if num_parents > 1:
            # must insert an aggregator node
            aggregator = Aggregator()  # all parents of self must be rerouted to this aggregatorNode
            # self.getInputNode().registerAggregator(aggregator)

            for parent in self.parents:
                aggregator.add_parent(parent)
                parent.children = [aggregator if x == self else x for x in
                                   parent.children]  # replaces self as a child with the new aggregator node

            self.parents = []  # none of the parents are now a parent to this node - instead only the aggregator is
            # self.parents.append(aggregator)
            aggregator.add_child(self)

            for previousParent in aggregator.parents:
                previousParent.insert_aggregator_nodes("fromTop")

        elif num_parents == 1:
            self.parents[0].insert_aggregator_nodes("fromTop")

        if self.is_output_node():
            input_node = self.get_input_node()
            input_node.clear()
            input_node.get_traversal_ids('_')

    def pass_ann_input_up_graph(self, input, parent_id="", configuration_run=False):
        """
        Called by the forward method of the NN - traverses the module graph passes the nn's input all the way up through
        the graph aggregator nodes wait for all inputs before passing outputs
        """
        if configuration_run and not (input is None):
            try:
                self.shape_layer(list(input.size()))
            except Exception as e:
                print(e)
                raise Exception("failed on " + repr(input.size()))

        output = self.pass_input_through_layer(input)  # will call aggregation if is aggregator node

        child_out = None
        for child in self.children:
            co = child.pass_ann_input_up_graph(output, self.traversal_id, configuration_run=configuration_run)
            if co is not None:
                child_out = co

        # if child_out is not None:
        #     return child_out

        if self.is_output_node():
            if output is None:
                print('Reached output and was none: error!')
            return output  # output of final output node must be bubbled back up to the top entry point of the nn

        return child_out

    def pass_input_through_layer(self, input):
        if input is None:
            return None
        if self.deep_layer is None:
            raise Exception("no deep layer, cannot pass input through layer", self)

        if not (self.reshape is None):
            input = self.reshape.shape(input)

        if self.is_conv2d() and list(input.size())[2] < minimum_conv_dim:
            xkernel, ykernel = self.deep_layer.kernel_size
            xkernel, ykernel = (xkernel - 1) // 2, (ykernel - 1) // 2
            input = F.pad(input=input, pad=(ykernel, ykernel, xkernel, xkernel), mode='constant',
                          value=0)

        if self.regularisation is None:
            output = self.deep_layer(input)
        else:
            output = self.regularisation(self.deep_layer(input))

        if not self.reduction is None:
            if type(self.reduction) == nn.MaxPool2d or type(self.reduction) == nn.MaxPool1d:
                # a reduction should only be done on inputs large enough to reduce
                if list(input.size())[2] > minimum_conv_dim:
                    output = self.reduction(self.activation(output))
            else:
                print("Error: reduction", self.reduction, " is not implemented")

        if self.dropout is not None:
            output = self.dropout(output)

        # print("conv dim size of output:",list(output.size())[2])
        if self.is_linear() or (self.is_conv2d() and list(output.size())[2] > minimum_conv_dim):
            return self.activation(output)
        else:
            # is conv layer - is small. needs padding
            xkernel, ykernel = self.deep_layer.kernel_size
            xkernel, ykernel = (xkernel - 1) // 2, (ykernel - 1) // 2

            return F.pad(input=self.activation(output), pad=(ykernel, ykernel, xkernel, xkernel), mode='constant',
                         value=0)

    def shape_layer(self, input_shape):
        """
        adds reshape nodes and creates layer on this node

        :param input_shape: a list form of the shape of the input which will be given to this node
        """
        input_flat_size = Utils.get_flat_number(sizes=input_shape)
        try:
            features = input_shape[1]
        except:
            raise Exception("could not extract features from", input_shape)

        if (self.is_conv2d()):
            # TODO non square conv dims
            conv_dim = int(math.pow(input_flat_size / features, 0.5))
            if not (math.pow(conv_dim, 2) * features == input_flat_size):
                raise Exception("error calculating conv dim from input flat size:", input_flat_size, " tried conv size",
                                conv_dim)

            output_shape = [input_shape[0], features, conv_dim, conv_dim]
            # print('adding convreshape node for', input_shape, "num features:",features, "out shape:",output_shape)

        if (self.is_linear()):
            features = input_flat_size
            output_shape = [input_shape[0], input_flat_size]
            # print('adding linear reshape node for', input_shape, "num features:",features, "out shape:",output_shape)

        self.create_layer(features)
        if input_shape == output_shape:
            return
        # print("using reshape node from",input_shape,"to",output_shape)

        self.reshape = ReshapeNode(input_shape, output_shape)

    def get_parameters(self, parametersDict, top=True):

        if self not in parametersDict:
            if self.deep_layer is None:
                raise Exception("no deep layer - ", self)

            myParams = self.deep_layer.parameters()
            parametersDict[self] = myParams

            for child in self.children:
                child.get_parameters(parametersDict, top=False)

            if self.is_input_node():
                # print("input node returned to from get parameters")
                params = None
                for param in parametersDict.values():

                    if params is None:
                        params = list(param)
                    else:
                        params.extend(list(param))
                return params

    def get_net_size(self):
        net_params = self.get_parameters({})
        return sum(p.numel() for p in net_params if p.requires_grad)

    def get_plot_colour(self, include_shape=True):
        # print("plotting agg node")
        if include_shape:
            if self.deep_layer is None:
                return "rs"
            if self.is_conv2d():
                return "go"
            elif self.is_linear():
                return "co"
        else:
            if self.deep_layer is None:
                return "red"
            if self.is_conv2d():
                return "yellow"
            elif self.is_linear():
                return "cyan"

    def get_layer_type_name(self):
        layer_type = self.module_NEAT_node.layer_type

        extras = "\nout features:" + repr(self.out_features)
        extras += "\n" + repr(self.regularisation).split("(")[0] if not (self.regularisation is None) else ""
        extras += "\n" + repr(self.reduction).split("(")[0] if not (self.reduction is None) else ""
        extras += "\n" + repr(self.dropout).split("(")[0] if not (self.dropout is None) else ""


        if layer_type() == nn.Conv2d:
            return "Conv" + extras

        elif layer_type() == nn.Linear:
            return "Linear" + extras
        else:
            print("layer type", layer_type(), "not implemented")

    def get_out_features(self, deep_layer=None):
        """:returns out_channels if deep_layer is a Conv2d | out_features if deep_layer is Linear
        :parameter deep_layer: if none - performs operation on this nodes deep_layer, else performs on the provided layer"""
        if deep_layer is None:
            deep_layer = self.deep_layer

        if type(deep_layer) == nn.Conv2d:
            num_features = deep_layer.out_channels
        elif type(deep_layer) == nn.Linear:
            num_features = deep_layer.out_features
        else:
            print("cannot extract num features from layer type:", type(deep_layer))
            return None

        return num_features

    def get_in_features(self, deep_layer=None):
        """:returns out_channels if deep_layer is a Conv2d | out_features if deep_layer is Linear
        :parameter deep_layer: if none - performs operation on this nodes deep_layer, else performs on the provided layer"""
        if deep_layer is None:
            deep_layer = self.deep_layer

        if type(deep_layer) == nn.Conv2d:
            num_features = deep_layer.in_channels
        elif type(deep_layer) == nn.Linear:
            num_features = deep_layer.in_features
        else:
            print("cannot extract num features from layer type:", type(deep_layer))
            return None

        return num_features

    def get_feature_tuple(self, deep_layer, new_input):
        """:returns channelsOut,x,y if Conv2D | out_features if Linear"""
        if type(deep_layer) == nn.Conv2d:
            return self.get_out_features(deep_layer=deep_layer), new_input.size()[2], new_input.size()[3]
        elif type(deep_layer) == nn.Linear:
            return self.get_out_features(deep_layer=deep_layer)
        else:
            print("have not implemented layer type", deep_layer)

    def is_linear(self):
        return self.module_NEAT_node.layer_type.get_value() == nn.Linear

    def is_conv2d(self):
        return self.module_NEAT_node.layer_type.get_value() == nn.Conv2d

    def get_first_feature_count(self, input):
        layer_type = self.module_NEAT_node.layer_type
        if layer_type() == nn.Conv2d:
            return list(input.size())[1]

        elif layer_type() == nn.Linear:
            return Utils.get_flat_number(input)
        else:
            print("layer type", layer_type(), "not implemented")

    def report_fitness(self,*fitnesses):
        if self.fitness_values is None or not self.fitness_values:
            self.fitness_values = [0 for _ in fitnesses]

        for i, fitness in enumerate(fitnesses):
            self.fitness_values[i] = fitness
