from src.Graph.Node import Node
from src.Module.ReshapeNode import ReshapeNode
from src.Utilities import Utils
import torch.nn as nn
import torch
import torch.nn.functional as F
import random
import math
from src.NeuralNetwork.Net import ModuleNet
import copy


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

        self.reshape = None# reshapes the input given before passing it through this nodes deeplayer

        self.module_NEAT_genome = module_genome
        self.module_NEAT_node = module_NEAT_node

        if not (module_NEAT_node is None):
            self.generate_module_node_from_gene()

    def generate_module_node_from_gene(self):
        self.out_features = self.module_NEAT_node.out_features.get_value()
        self.activation = self.module_NEAT_node.activation.get_value()

        neat_regularisation = self.module_NEAT_node.layer_type.get_sub_value("regularisation", return_mutagen=True)
        neat_reduction = self.module_NEAT_node.layer_type.get_sub_value("reduction", return_mutagen=True)

        if not (neat_regularisation.get_value() is None):
            self.regularisation = neat_regularisation.get_value()(self.out_features)

        if not (neat_reduction.get_value() is None):
            if neat_reduction.get_value() == nn.MaxPool2d or neat_reduction.get_value() == nn.MaxPool1d:
                pool_size = neat_reduction.get_sub_value("pool_size")
                if neat_reduction.get_value() == nn.MaxPool2d:
                    self.reduction = nn.MaxPool2d(pool_size, pool_size)
                if neat_reduction.get_value() == nn.MaxPool1d:
                    self.reduction = nn.MaxPool1d(pool_size)
            else:
                print("Error not implemented reduction ", neat_reduction.get_value())

    def to_nn(self, in_features, device, print_graphs=False):
        self.create_layers(in_features=in_features, device=device)
        if print_graphs:
            self.plot_tree()
        return ModuleNet(self).to(device)

    def create_layers(self, in_features=None, device=torch.device("cpu")):
        if self.deep_layer is None or True:

            """decide in features"""
            if in_features is None:
                self.in_features = self.parents[
                    0].out_features  # only aggregator nodes should have more than one parent
            else:
                self.in_features = in_features

            layer_type = self.module_NEAT_node.layer_type
            if layer_type.get_value() == nn.Conv2d:
                try:
                    self.deep_layer = nn.Conv2d(self.in_features, self.out_features,
                                                kernel_size=layer_type.get_sub_value("conv_window_size"),
                                                stride=layer_type.get_sub_value("conv_stride"))
                    try:
                        self.deep_layer = self.deep_layer.to(device)
                    except:
                        print("created conv layer - but failed to move it to device", device)

                except Exception as e:
                    print("Error:", e)
                    print("failed to create conv", self, self.in_features, self.out_features,
                          layer_type.get_sub_value("conv_window_size"),
                          layer_type.get_sub_value("conv_stride"), "deep layer:", self.deep_layer)
                    print('Module with error', self.module_NEAT_genome.connections)

            else:
                self.deep_layer = layer_type.get_value()(self.in_features, self.out_features).to(device)

            if not (self.reduction is None):
                self.reduction = self.reduction.to(device)

            if not (self.regularisation is None):
                self.regularisation = self.regularisation.to(device)

            for child in self.children:
                child.create_layers(device=device)

        else:
            print("already has deep layers", self.is_input_node())

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

    def pass_ann_input_up_graph(self, input, parent_id=""):
        """
        Called by the forward method of the NN - traverses the module graph passes the nn's input all the way up through
        the graph aggregator nodes wait for all inputs before passing outputs
        """
        output = self.pass_input_through_layer(input)  # will call aggregation if is aggregator node

        child_out = None
        for child in self.children:
            co = child.pass_ann_input_up_graph(output, self.traversalID)
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

        if not (self.reshape is None):
            if not (self.is_input_node()):
                print("non input node with a reshape node")
            input = self.reshape.shape(input)

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

        #print("conv dim size of output:",list(output.size())[2])
        if self.is_linear() or (self.is_conv2d() and list(output.size())[2] > minimum_conv_dim):
            return self.activation(output)
        else:
            # is conv layer - is small. needs padding
            xkernel, ykernel = self.deep_layer.kernel_size
            xkernel, ykernel = (xkernel - 1) // 2, (ykernel - 1) // 2

            return F.pad(input=self.activation(output), pad=(ykernel, ykernel, xkernel, xkernel), mode='constant',
                         value=0)

    def add_reshape_node(self, input_shape):
        features = self.get_in_features()
        input_flat_size = Utils.get_flat_number(sizes=input_shape)

        if(self.is_conv2d()):
            #TODO non square conv dims
            conv_dim = int(math.pow(input_flat_size,0.5))
            if not (math.pow(conv_dim,2) == input_flat_size):
                print("error calculating conv dim from input flat size:",input_flat_size, " tried conv size",conv_dim)
                return
            output_shape = [input_shape[0], features, conv_dim,conv_dim]
            #print('adding convreshape node for', input_shape, "num features:",features, "out shape:",output_shape)

        if(self.is_linear()):
            output_shape = [input_shape[0], input_flat_size]
            #print('adding linear reshape node for', input_shape, "num features:",features, "out shape:",output_shape)

        if input_shape == output_shape:
            return

        self.reshape = ReshapeNode(input_shape, output_shape)

    def get_parameters(self, parametersDict, top=True):

        if self not in parametersDict:
            if (self.deep_layer is None):
                print("no deep layer - ", self)

            # if(top ):
            #     print("top is input:",self.is_input_node())
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
                        params += list(param)
                return params

    def print_node(self, print_to_console=True):
        out = " " * (len(self.traversalID)) + self.traversalID
        if print_to_console:
            print(out)
        else:
            return out

    def get_plot_colour(self):
        # print("plotting agg node")
        if self.deep_layer is None:
            return "rs"
        if self.is_conv2d():
            return "go"
        elif self.is_linear():
            return "co"

    # def get_dimensionality(self):
    #     print("need to implement get dimensionality")
    #     # 10*10 because by the time the 28*28 has gone through all the convs - it has been reduced to 10810
    #     return 10 * 10 * self.deep_layer.out_channels

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
            print("have not implemented layer type",deep_layer)
    def is_linear(self):
        return type(self.deep_layer) == nn.Linear

    def is_conv2d(self):
        return type(self.deep_layer) == nn.Conv2d

    def get_first_feature_count(self, input):
        layer_type = self.module_NEAT_node.layer_type
        if layer_type.get_value() == nn.Conv2d:
            return list(input.size())[1]

        elif layer_type.get_value() == nn.Linear:
            return Utils.get_flat_number(input)
        else:
            print("layer type",layer_type.get_value(),"not implemented")
