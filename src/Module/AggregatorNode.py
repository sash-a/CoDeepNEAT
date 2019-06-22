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
        for parent in self.module_node_input_ids:
            deep_layer = parent.deepLayer
            new_input = self.accountedForInputIDs[parent.traversalID]
            # combine inputs

            if previous_features == -1:
                # first input. no issue by default
                input_type = type(deep_layer)

            else:
                #second input and onward
                if type(deep_layer) == input_type:
                    # same layer type as seen up till now
                    new_features = self.get_feature_tuple(deep_layer, new_input)

                    if new_features == previous_features:
                        # no issue
                        pass
                    else:
                        # different input shapes
                        if input_type == nn.Conv2d:
                            # print("merging conv layers")
                            new_input, previous_inputs = AggregatorOperations.merge_conv_outputs(previous_features,previous_inputs,new_features, new_input)

                        elif input_type == nn.Linear:
                            new_input, previous_inputs = AggregatorOperations.merge_linear_outputs(previous_inputs, new_input)
                        else:
                            print("not yet implemented merge of layer type:", input_type)
                else:
                    print("trying to merge layers of different types:", type(deep_layer), ";", input_type,
                          "this has not been implemented yet")

            previous_features = self.get_feature_tuple(deep_layer, previous_inputs[0])
            if(not new_input is None):
                previous_inputs.append(new_input)
            input_shapes += "," + repr(new_input.size())

        output = torch.sum(torch.stack(previous_inputs), dim=0)

        return output




