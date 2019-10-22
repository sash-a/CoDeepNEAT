from torch import nn

import src.Validation.DataLoader as DL
from src.Validation.Validation import get_accuracy_estimate_for_network
from src.CoDeepNEAT.CDNGenomes.BlueprintGenome import BlueprintGenome
from src.Config import Config
from src.NEAT.Gene import ConnectionGene
from src.CoDeepNEAT.CDNGenomes.ModuleGenome import ModuleGenome
from src.CoDeepNEAT.CDNNodes.BlueprintNode import BlueprintNEATNode
from src.CoDeepNEAT.CDNNodes.ModuleNode import ModuleNEATNode, NodeType
from src.NEAT.Species import Species
from src.Phenotype2.NeuralNetwork import Network

conn0 = ConnectionGene(0, 0, 2)
conn1 = ConnectionGene(1, 0, 3)
conn2 = ConnectionGene(2, 2, 5)
conn3 = ConnectionGene(3, 3, 6)
conn4 = ConnectionGene(4, 6, 4)
conn5 = ConnectionGene(5, 5, 1)
conn6 = ConnectionGene(6, 4, 1)
conn7 = ConnectionGene(7, 3, 5)

# conn3.enabled.set_value(False)

n0 = ModuleNEATNode(0, NodeType.INPUT)
n1 = ModuleNEATNode(1, NodeType.OUTPUT)
n2 = ModuleNEATNode(2)
n3 = ModuleNEATNode(3)
n4 = ModuleNEATNode(4)
n5 = ModuleNEATNode(5)
n6 = ModuleNEATNode(6)
genome0 = ModuleGenome([conn0, conn1, conn2, conn3, conn4, conn5, conn6, conn7], [n0, n1, n2, n3, n4, n5, n6])

conn10 = ConnectionGene(0, 0, 1)
n10 = ModuleNEATNode(0, NodeType.INPUT)
n11 = ModuleNEATNode(1, NodeType.OUTPUT)
n10.layer_type.set_value(nn.Linear)
genome1 = ModuleGenome([conn10], [n10, n11])

spcs = [Species(genome0), Species(genome1)]

bn0 = BlueprintNEATNode(0, NodeType.INPUT)
bn1 = BlueprintNEATNode(1, NodeType.OUTPUT)

bn0.species_number.set_value(0)
bn1.species_number.set_value(1)

bpg = BlueprintGenome([conn10], [bn0, bn1])

Config.use_graph = True
x, target = DL.sample_data(Config.get_device(), 256)
enen = Network(bpg, spcs, list(x.shape)).to(Config.get_device())
enen.visualize()

# get_accuracy_estimate_for_network(enen)
