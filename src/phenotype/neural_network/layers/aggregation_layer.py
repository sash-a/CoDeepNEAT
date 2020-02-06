from typing import List

import torch
from torch import tensor, nn

from src.configuration import config
from src.phenotype.neural_network.aggregation.merger import merge
from src.phenotype.neural_network.layers.base_layer import BaseLayer


class AggregationLayer(BaseLayer):
    def __init__(self, num_inputs: int, name: str, lossy: bool, try_output_conv: bool):
        """
        :param lossy: choice between summing or catting to merge
        :param try_output_conv: determines whether the agg layer outputs a conv or linear
            in the case where both are provided as inputs
        """
        super().__init__(name)

        self.n_inputs_expected: int = num_inputs
        self.n_inputs_received: int = 0
        self.channel_resizers: nn.ModuleList = nn.ModuleList()
        self.inputs: List[tensor] = []

        self.lossy: bool = lossy
        self.try_output_conv: bool = try_output_conv

    def forward(self, input):
        self.n_inputs_received += 1
        self.inputs.append(input)

        if self.n_inputs_received > self.n_inputs_expected:
            raise Exception('Received more inputs than expected')
        elif self.n_inputs_received < self.n_inputs_expected:
            # waiting on other inputs
            return

        if config.old_agg:
            aggregated = self.old_aggregate()
        else:
            aggregated = merge(self.inputs, self)
        self.reset()
        return aggregated

    def reset(self):
        self.n_inputs_received = 0
        self.inputs = []

    def create_layer(self, in_shape: list):
        # print('creating agg layer')
        self.n_inputs_received += 1
        self.inputs.append(torch.zeros(in_shape).to(config.get_device()))

        if self.n_inputs_received > self.n_inputs_expected:
            raise Exception('Received more inputs than expected')
        elif self.n_inputs_received < self.n_inputs_expected:
            return

        # Calculate the output shape of the layer by passing input through it
        # self.out_shape = list(self.aggregate().size())
        if config.old_agg:
            self.out_shape = list(self.old_aggregate().size())
        else:
            self.out_shape = list(merge(self.inputs, self).size())
        self.reset()

        return self.out_shape

    def get_layer_info(self):
        return 'aggregation layer' \
               '\nLossy: ' + str(self.lossy) + \
               '\nConv agg: ' + str(self.try_output_conv)

    def homogenise_outputs_list(self, outputs, homogeniser):
        """
        loops through the inputs, building up a list of homogeneously sized transformed inputs
        for each new input, either reshapes that input, or the homogeneous list

        :param outputs: full list of unhomogenous output tensors - of the same layer type (dimensionality)
        :param homogeniser: the function used to return the homogenous list for this layer type
        :param outputs_deep_layers: a map of output tensor to deep layer it came from
        :return: a homogenous list of tensors
        """
        if outputs is None:
            raise Exception("Error: trying to homogenise null outputs list")

        length_of_outputs = len(outputs)
        i = 0
        while i < length_of_outputs:
            # all list items from 0:i-1 are homogenous

            homogenous_features = outputs[0].size()
            new_features = outputs[i].size()

            if new_features != homogenous_features:
                """either the list up till this point or the new  input needs modification"""

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
