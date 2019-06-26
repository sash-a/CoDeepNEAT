from src.Graph.Node import Node
import torch.nn as nn
import torch
import torch.nn.functional as F
import random

#random.seed(0)


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

    def __init__(self, device=torch.device("cpu")):
        Node.__init__(self)
        self.deepLayer = None  # an nn layer object such as    nn.Conv2d(3, 6, 5) or nn.Linear(84, 10)
        self.inFeatures = -1
        self.outFeatures = -1
        self.traversalID = ""
        self.activation = F.relu
        if random.randint(0, 0) == 0:
            self.reduction = nn.MaxPool2d(2, 2)
            pass
        else:
            self.reduction = None
        self.regularisation = None

    def create_layers(self, in_features=None, out_features=25, device=torch.device("cpu")):
        self.outFeatures = out_features
        if self.deepLayer is None:
            if in_features is None:
                self.inFeatures = self.parents[0].outFeatures  # only aggregator nodes should have more than one parent
            else:
                self.inFeatures = in_features
            self.deepLayer = nn.Conv2d(self.inFeatures, self.outFeatures, 3, 1).to(device)
            if random.randint(0, 1) == 0:
                self.regularisation = nn.BatchNorm2d(self.outFeatures).to(device)
                pass

            for child in self.children:
                child.create_layers(device=device)

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
            return output  # output of final output node must be bubbled back up to the top entry point of the nn

    def pass_input_through_layer(self, input):
        if input is None:
            return None

        if self.regularisation is None:
            output = self.deepLayer(input)
        else:
            output = self.regularisation(self.deepLayer(input))

        if not self.reduction is None:
            if type(self.reduction) == nn.MaxPool2d:
                # a reduction should only be done on inputs large enough to reduce
                if list(input.size())[2] > 5:
                    return self.reduction(self.activation(output))
            else:
                return self.reduction(self.activation(output))

        if type(self.deepLayer) == nn.Linear or (type(self.deepLayer) == nn.Conv2d and list(input.size())[2] > 5):
            return self.activation(output)
        else:
            # is conv layer - is small. needs padding
            xkernel, ykernel = self.deepLayer.kernel_size
            xkernel, ykernel = (xkernel - 1) // 2, (ykernel - 1) // 2
            return F.pad(input=self.activation(output), pad=(ykernel, ykernel, xkernel, xkernel), mode='constant',
                         value=0)

    def get_parameters(self, parametersDict):
        if self not in parametersDict:
            myParams = self.deepLayer.parameters()
            parametersDict[self] = myParams

            for child in self.children:
                child.get_parameters(parametersDict)

            if self.is_input_node():
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
        if (self.deepLayer is None):
            return "rs"
        if (type(self.deepLayer) == nn.Conv2d):
            return "go"
        elif (type(self.deepLayer) == nn.Linear):
            return "co"

    def get_dimensionality(self):
        print("need to implement get dimensionality")
        # 10*10 because by the time the 28*28 has gone through all the convs - it has been reduced to 10810
        return 10 * 10 * self.deepLayer.out_channels

    def get_out_features(self, deep_layer=None):
        """:returns out_channels if deep_layer is a Conv2d | out_features if deep_layer is Linear
        :parameter deep_layer: if none - performs operation on this nodes deep_layer, else performs on the provided layer"""
        if deep_layer is None:
            deep_layer = self.deepLayer

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
        if(type(deep_layer) == nn.Conv2d):
            return self.get_out_features(deep_layer=deep_layer), new_input.size()[2], new_input.size()[3]
        elif(type(deep_layer) == nn.Linear):
            return self.get_out_features(deep_layer=deep_layer)
