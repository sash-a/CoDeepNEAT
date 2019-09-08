import torch
import torch.nn as nn

from src.Config import Config
from src.Phenotype import AggregatorOperations
from src.Phenotype.ModuleNode import ModuleNode as Module


class AggregatorNode(Module):
    aggregationType = ""
    module_node_input_ids = []  # a list of the ID's of all the modules which pass their output to this aggregator
    accountedForInputIDs = {}  # map from ID's to input vectors

    def __init__(self):
        Module.__init__(self, None, None)
        self.module_node_input_ids = []
        self.accountedForInputIDs = {}
        self.out_features = 25

    def create_layer(self, in_features=None, device=torch.device("cpu")):
        """passes this recursive action onwards"""
        for child in self.children:
            child.create_layer(device=device)

    def generate_module_node_from_gene(self, feature_multiplier=1):
        pass

    def delete_layer(self):
        pass

    def insert_aggregator_nodes(self, state="start"):
        # if aggregator node has already been created - then the multi parent situation has already been dealt with here
        # since at the time the decendants of this aggregator node were travered further already, there is no need to
        # traverse its decendants again
        pass

    def get_parameters(self, parameters_dict, top=False):
        for child in self.children:
            child.get_parameters(parameters_dict, top=False)

    def add_parent(self, parent):
        self.module_node_input_ids.append(parent)
        self.parents.append(parent)

    def reset_node(self):
        """called between uses of the forward method of the NN created.
            tells the aggregator a new pass is underway and all inputs must again be waited for to pass forward
        """
        self.accountedForInputIDs = {}

    def get_plot_colour(self, include_shape=True):
        if include_shape:
            return 'bo'
        else:
            return "violet"

    def get_layer_type_name(self):
        return "Aggregator"

    def pass_ann_input_up_graph(self, input, parent_id="", configuration_run=False):
        """collects the inputs which are passed to it one at a time"""

        self.accountedForInputIDs[parent_id] = input

        if len(self.module_node_input_ids) == len(self.accountedForInputIDs):
            # all inputs have arrived
            # may now aggregate and pass upwards
            out = super(AggregatorNode, self).pass_ann_input_up_graph(None, configuration_run=configuration_run)
            self.reset_node()

            return out

    def pass_input_through_layer(self, _):
        """aggregates inputs collected and passes the result up the graph"""
        conv_outputs = []
        linear_outputs = []
        outputs_deep_layers = {}

        has_linear = False
        has_conv = False

        for parent in self.module_node_input_ids:
            # separte inputs by typee
            deep_layer = parent.deep_layer
            new_input = self.accountedForInputIDs[parent.traversal_id]
            outputs_deep_layers[new_input] = deep_layer
            if type(deep_layer) == nn.Conv2d:
                conv_outputs.append(new_input)
                has_conv = True
            elif type(deep_layer) == nn.Linear:
                linear_outputs.append(new_input)
                has_linear = True

        if not (not has_linear or has_conv):
            linear_outputs = self.homogenise_outputs_list(linear_outputs, AggregatorOperations.merge_linear_outputs,
                                                          outputs_deep_layers)
            return torch.sum(torch.stack(linear_outputs), dim=0)
        elif has_conv and not has_linear:
            conv_outputs = self.homogenise_outputs_list(conv_outputs, AggregatorOperations.merge_conv_outputs,
                                                        outputs_deep_layers)
            if conv_outputs is None:
                print("Error: null conv outputs returned from homogeniser")
            try:
                return torch.sum(torch.stack(conv_outputs), dim=0)
            except Exception as e:
                print("failed to sum non homogenous inputs: ", end="")
                for conv in conv_outputs:
                    print(conv.size(), end=",")
                raise e
        elif has_linear and has_conv:
            linear_outputs = self.homogenise_outputs_list(linear_outputs, AggregatorOperations.merge_linear_outputs,
                                                          outputs_deep_layers)
            conv_outputs = self.homogenise_outputs_list(conv_outputs, AggregatorOperations.merge_conv_outputs,
                                                        outputs_deep_layers)
            return AggregatorOperations.merge_linear_and_conv(torch.sum(torch.stack(linear_outputs), dim=0),
                                                              torch.sum(torch.stack(conv_outputs), dim=0))
        else:
            print("error - agg node received neither conv or linear inputs")
            return None

    def homogenise_outputs_list(self, outputs, homogeniser, outputs_deep_layers):
        """

        :param outputs: full list of unhomogenous output tensors - of the same layer type (dimensionality)
        :param homogeniser: the function used to return the homogenous list for this layer type
        :param outputs_deep_layers: a map of output tensor to deep layer it came from
        :return: a homogenous list of tensors
        """
        if outputs is None:
            raise Exception("Error: trying to homogenise null outputs list")

        if len(outputs) > len(outputs_deep_layers):
            raise Exception("outputs list(", len(outputs), ") and outputs map (out->layer)(", len(outputs_deep_layers),
                            ") do not match size")

        homogenous_features = None
        length_of_outputs = len(outputs)
        i = 0
        while i < length_of_outputs:
            # all list items from 0:i-1 are homogenous

            homogenous_features = outputs[0].size()
            new_features = outputs[i].size()

            if new_features != homogenous_features:
                # either the list up till this point or the new  input needs modification
                if i < length_of_outputs - 1:
                    outputs = homogeniser(outputs[:i], outputs[i]) + outputs[i + 1:]
                else:  # i== len - 1
                    outputs = homogeniser(outputs[:i], outputs[i])

                if outputs is None:
                    raise Exception("null outputs returned from homogeniser")
                if len(outputs) < length_of_outputs:
                    """homogeniser can sometimes collapse the previous inputs into one in certain circumstances"""
                    change = length_of_outputs - len(outputs)
                    i = 0
                    length_of_outputs = len(outputs)

            i += 1

        if outputs is None:
            raise Exception("Error: outputs turned null from homogenising using" + repr(homogeniser))

        return outputs
