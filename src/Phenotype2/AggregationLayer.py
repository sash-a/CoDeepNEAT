import torch
from torch import tensor
from typing import List

from src.Config import Config
from src.Phenotype2.LayerUtils import BaseLayer
from src.Phenotype import AggregatorOperations


class AggregationLayer(BaseLayer):
    def __init__(self, num_inputs: int, name):
        super().__init__(name)

        self.n_inputs_expected: int = num_inputs
        self.n_inputs_received: int = 0
        self.inputs: List[tensor] = []

    def forward(self, input):
        self.n_inputs_received += 1
        self.inputs.append(input)

        if self.n_inputs_received > self.n_inputs_expected:
            raise Exception('Received more inputs than expected')
        elif self.n_inputs_received < self.n_inputs_expected:
            return

        aggregated = self.aggregate()
        # TODO: inputs should probably be reset after the step although can be pretty sure that this is fine
        self.reset()
        return aggregated

    def reset(self):
        self.n_inputs_received = 0
        self.inputs = []

    def create_layer(self, in_shape: list):
        self.n_inputs_received += 1
        self.inputs.append(torch.zeros(in_shape).to(Config.get_device()))

        if self.n_inputs_received > self.n_inputs_expected:
            raise Exception('Received more inputs than expected')
        elif self.n_inputs_received < self.n_inputs_expected:
            return

        # Calculate the output shape of the layer by passing input through it
        self.out_shape = list(self.aggregate().size())
        self.reset()

        return self.out_shape

    def get_layer_info(self):
        return "Aggregation layer"

    def aggregate(self):
        input_dims = list(map(lambda x: len(list(x.size())), self.inputs))
        has_linear = 2 in input_dims
        has_conv = 4 in input_dims

        linear_inputs = [x for x in self.inputs if len(list(x.size())) == 2]
        conv_inputs = [x for x in self.inputs if len(list(x.size())) == 4]

        if not (not has_linear or has_conv):
            linear_inputs = self.homogenise_outputs_list(linear_inputs, AggregatorOperations.merge_linear_outputs)
            return torch.sum(torch.stack(linear_inputs), dim=0)
        elif has_conv and not has_linear:
            conv_inputs = self.homogenise_outputs_list(conv_inputs, AggregatorOperations.merge_conv_outputs)
            if conv_inputs is None:
                print("Error: null conv outputs returned from homogeniser")
            try:
                return torch.sum(torch.stack(conv_inputs), dim=0)
            except Exception as e:
                print("failed to sum non homogenous inputs: ", end="")
                for conv in conv_inputs:
                    print(conv.size(), end=",")
                raise e
        elif has_linear and has_conv:
            linear_inputs = self.homogenise_outputs_list(linear_inputs, AggregatorOperations.merge_linear_outputs)
            conv_inputs = self.homogenise_outputs_list(conv_inputs, AggregatorOperations.merge_conv_outputs)
            return AggregatorOperations.merge_linear_and_conv(torch.sum(torch.stack(linear_inputs), dim=0),
                                                              torch.sum(torch.stack(conv_inputs), dim=0))
        else:
            print("error - agg node received neither conv or linear inputs")
            return None

    def homogenise_outputs_list(self, outputs, homogeniser, outputs_deep_layers=None):
        """
        :param outputs: full list of unhomogenous output tensors - of the same layer type (dimensionality)
        :param homogeniser: the function used to return the homogenous list for this layer type
        :param outputs_deep_layers: a map of output tensor to deep layer it came from
        :return: a homogenous list of tensors
        """
        if outputs is None:
            raise Exception("Error: trying to homogenise null outputs list")

        # if len(outputs) > len(outputs_deep_layers):
        #     raise Exception("outputs list(", len(outputs), ") and outputs map (out->layer)(", len(outputs_deep_layers),
        #                     ") do not match size")

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
