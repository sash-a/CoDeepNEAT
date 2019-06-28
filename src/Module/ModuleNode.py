from src.Graph.Node import Node
import torch.nn as nn
import torch
import torch.nn.functional as F
import random
from src.NeuralNetwork.Net import ModuleNet


# random.seed(0)


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
        self.blueprint_connections = []#blueprints connect module nodes vai aprent/child relationships - these must be recored so they may be severed when the blueprint is dissolved

        self.deep_layer = None  # an nn layer object such as    nn.Conv2d(3, 6, 5) or nn.Linear(84, 10)
        self.in_features = -1
        self.out_features = -1
        self.activation = None

        self.reduction = None
        self.regularisation = None

        self.module_NEAT_genome = module_genome
        self.module_NEAT_node = module_NEAT_node

        if not (module_NEAT_node is None):
            self.generate_module_node_from_gene()

    def generate_module_node_from_gene(self ):
        try:
            self.out_features = self.module_NEAT_node.out_features.get_value()
        except:
            print("no out features attached to",type(self.module_NEAT_node))

        self.activation = self.module_NEAT_node.activation.get_value()

    def to_nn(self, in_features, device, print_graphs=False):
        self.create_layers(in_features=in_features, device=device)
        self.insert_aggregator_nodes()
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
                                                stride=layer_type.get_sub_value("conv_stride")).to(device)
                except:
                    print("failed to create conv", self, self.in_features, self.out_features,
                          layer_type.get_sub_value("conv_window_size"),
                          layer_type.get_sub_value("conv_stride"))
            else:
                self.deep_layer = layer_type.get_value()(self.in_features, self.out_features).to(device)

            if not (self.module_NEAT_node.regularisation.get_value()) is None:
                self.regularisation = self.module_NEAT_node.regularisation.get_value()(self.out_features).to(device)

            if not (self.module_NEAT_node.reduction.get_value() is None):
                reduction = self.module_NEAT_node.reduction
                if reduction.get_value() == nn.MaxPool2d:
                    pool_size = reduction.get_sub_value("pool_size")
                    self.reduction = nn.MaxPool2d(pool_size, pool_size).to(device)
                else:
                    self.reduction = reduction.get_value()().to(device)

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
            self.get_input_node().get_traversal_ids('_')

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

        if child_out is not None:
            return child_out

        if self.is_output_node():
            #print("output node reached in module graph - returning output == none ~", (output is None))
            return output  # output of final output node must be bubbled back up to the top entry point of the nn

    def pass_input_through_layer(self, input):
        if input is None:
            return None

        if self.regularisation is None:
            output = self.deep_layer(input)
        else:
            output = self.regularisation(self.deep_layer(input))

        if not self.reduction is None:
            if type(self.reduction) == nn.MaxPool2d:
                # a reduction should only be done on inputs large enough to reduce
                if list(input.size())[2] > 5:
                    return self.reduction(self.activation(output))
            else:
                return self.reduction(self.activation(output))

        if type(self.deep_layer) == nn.Linear or (type(self.deep_layer) == nn.Conv2d and list(input.size())[2] > 5):
            return self.activation(output)
        else:
            # is conv layer - is small. needs padding
            xkernel, ykernel = self.deep_layer.kernel_size
            xkernel, ykernel = (xkernel - 1) // 2, (ykernel - 1) // 2
            return F.pad(input=self.activation(output), pad=(ykernel, ykernel, xkernel, xkernel), mode='constant',
                         value=0)

    def get_parameters(self, parametersDict, top = True):

        if self not in parametersDict:
            if (self.deep_layer is None):
                print("no deep layer - ", self)

            # if(top ):
            #     print("top is input:",self.is_input_node())
            myParams = self.deep_layer.parameters()
            parametersDict[self] = myParams

            for child in self.children:
                child.get_parameters(parametersDict, top = False)

            if self.is_input_node():
                #print("input node returned to from get parameters")
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
        if type(self.deep_layer) == nn.Conv2d:
            return "go"
        elif type(self.deep_layer) == nn.Linear:
            return "co"

    def get_dimensionality(self):
        print("need to implement get dimensionality")
        # 10*10 because by the time the 28*28 has gone through all the convs - it has been reduced to 10810
        return 10 * 10 * self.deep_layer.out_channels

    def get_out_features(self, deep_layer=None):
        """:returns out_channels if deep_layer is a Conv2d | out_features if deep_layer is Linear
        :parameter deep_layer: if none - performs operation on this nodes deep_layer, else performs on the provided layer"""
        if deep_layer is None:
            deep_layer = self.deep_layer

        if type(deep_layer) == nn.Conv2d:
            num_features = deep_layer.out_channels
        elif type(deep_layer) == nn.Linear:
            num_features = deep_layer.deepLayer.out_features
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

    def add_child(self, child_node, connection_type_is_module = True):
        super(ModuleNode,self).add_child(child_node,connection_type_is_module)
        if(not connection_type_is_module):
            self.blueprint_connections.append(child_node)

    def clear(self):
        for blueprint_connection in self.blueprint_connections:
            self.remove_child(blueprint_connection)

        self.blueprint_connections = []
        for child in self.children:
            child.clear()
