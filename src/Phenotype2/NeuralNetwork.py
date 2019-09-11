from torch import nn
from src.CoDeepNEAT.CDNGenomes import BlueprintGenome

from src.Phenotype2.Layer import Layer
from src.Phenotype2.AggregationLayer import AggregationLayer

from typing import Type


class Network(nn.Module):
    def __init__(self, blueprint: BlueprintGenome, output_dim=10):
        super().__init__()
        self.blueprint: BlueprintGenome = blueprint

        # TODO: add children

        self.final_layer = nn.Linear(1, output_dim)  # TODO

        # TODO: get params and add optimizer

    def forward(self, input):
        pass


from NEAT.Gene import ConnectionGene
from CoDeepNEAT.CDNGenomes import ModuleGenome
from CoDeepNEAT.CDNNodes import ModulenNEATNode, NodeType

conn0 = ConnectionGene(0, 0, 2)
conn1 = ConnectionGene(1, 0, 3)
conn2 = ConnectionGene(2, 2, 5)
conn3 = ConnectionGene(3, 3, 6)
conn4 = ConnectionGene(4, 6, 4)
conn5 = ConnectionGene(5, 5, 1)
conn6 = ConnectionGene(6, 4, 1)
conn7 = ConnectionGene(7, 3, 5)

# conn3.enabled.set_value(False)

n0 = ModulenNEATNode(0, NodeType.INPUT)
n1 = ModulenNEATNode(1, NodeType.OUTPUT)
n2 = ModulenNEATNode(2)
n3 = ModulenNEATNode(3)
n4 = ModulenNEATNode(4)
n5 = ModulenNEATNode(5)
n6 = ModulenNEATNode(6)

genome = ModuleGenome([conn0, conn1, conn2, conn3, conn4, conn5, conn6, conn7], [n0, n1, n2, n3, n4, n5, n6])

lay = genome.to_phenotype(None)

print()
print(lay)
