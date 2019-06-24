from src.Module.ModuleNode import ModuleNode as Module
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.Module import AggregatorOperations


class AggregatorNode(Module):
    aggregationType = ""
    module_node_input_ids = []  # a list of the ID's of all the modules which pass their output to this aggregator
    accountedForInputIDs = {}  # map from ID's to input vectors

    def __init__(self):
        Module.__init__(self)
        self.module_node_input_ids = []
        self.accountedForInputIDs = {}

    def insert_aggregator_nodes(self, state="start"):
        # if aggregator node has already been created - then the multi parent situation has already been dealt with here
        # since at the time the decendants of this aggregator node were travered further already, there is no need to
        # traverse its decendants again
        pass

    def get_parameters(self, parameters_dict):
        for child in self.children:
            child.get_parameters(parameters_dict)

    def add_parent(self, parent):
        self.module_node_input_ids.append(parent)
        self.parents.append(parent)

    def reset_node(self):
        """typically called between uses of the forward method of the NN created
            tells the aggregator a new pass is underway and all inputs must again be waited for to pass forward
        """
        self.accountedForInputIDs = {}

    def get_plot_colour(self):
        return 'bo'


    def pass_ann_input_up_graph(self, input, parent_id=""):
        self.accountedForInputIDs[parent_id] = input

        if len(self.module_node_input_ids) == len(self.accountedForInputIDs):
            # all inputs have arrived
            # may now aggregate and pass upwards
            out = super(AggregatorNode, self).pass_ann_input_up_graph(None)
            # if(not out is None):
            #     print("agg got non null out")
            self.reset_node()

            return out

    def pass_input_through_layer(self, _):
        output = None
        previous_inputs = []  # method ensures that previous_inputs is always homogenous as new inputs are added
        input_type = None
        previous_features = -1

        input_shapes = ""

        conv_outputs = []
        linear_outputs = []
        outputs_deep_layers = {}

        for parent in self.module_node_input_ids:
            #separte inputs by typee
            deep_layer = parent.deepLayer
            new_input = self.accountedForInputIDs[parent.traversalID]
            outputs_deep_layers[new_input] = deep_layer
            if(type(deep_layer) == nn.Conv2d):
                conv_outputs.append(new_input)
            elif(type(deep_layer) == nn.Linear):
                linear_outputs.append(new_input)

        conv_outputs = self.homogenise_outputs_list(conv_outputs, AggregatorOperations.merge_conv_outputs,outputs_deep_layers)
        linear_outputs= self.homogenise_outputs_list(linear_outputs, AggregatorOperations.merge_linear_outputs,outputs_deep_layers)

        output = torch.sum(torch.stack(conv_outputs), dim=0)

        return output

    def homogenise_outputs_list(self, outputs, homogeniser, outputs_deep_layers):
        """

        :param outputs: full list of unhomogenous output tensors - of the same layer type (dimensionality)
        :param homogeniser: the function used to return the homogenous list for this layer type
        :param outputs_deep_layers: a map of output tensor to deep layer it came from
        :return: a homogenous list of tensors
        """
        homogenous_conv_features = None
        for i in range(len(outputs)):
            #all list items from 0:i-1 are homogenous
            conv_layer = outputs_deep_layers[outputs[i]]
            if (homogenous_conv_features is None):
                homogenous_conv_features = self.get_feature_tuple(conv_layer, outputs[i])
                #print("setting hom features to:",homogenous_conv_features, "num outs:",len(outputs))
            else:
                new_conv_features = self.get_feature_tuple(conv_layer, outputs[i])
                if (not new_conv_features == homogenous_conv_features):
                    # either the list up till this point or the new  input needs modification
                    if(len(outputs)>i+1):
                        outputs = homogeniser(homogenous_conv_features,outputs[:i], new_conv_features,outputs[i]) + outputs[i+1:]
                    else:
                        outputs = homogeniser(homogenous_conv_features,outputs[:i], new_conv_features,outputs[i])

                    #print("hom shape:",outputs[0].size(),"new shape:",outputs[i].size(), "i:",i)

        return outputs


