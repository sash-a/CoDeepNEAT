from torch import nn, tensor
from typing import List
from src.Phenotype2.LayerUtils import BaseLayer


class AggregationLayer(BaseLayer):
    def __init__(self, num_inputs: int):
        super().__init__()
        self.num_inputs: int = num_inputs

        self.received_inputs = 0
        self.inputs: List[tensor] = []

    def forward(self, input):
        self.received_inputs += 1
        self.inputs.append(input)

        if self.received_inputs > self.num_inputs:
            raise Exception('Received more inputs than expected')
        elif self.received_inputs < self.num_inputs:
            return

        # TODO: should received inputs be reset here? Otherwise need to reset them somewhere.
        return self.aggregate()

    def aggregate(self):
        pass
