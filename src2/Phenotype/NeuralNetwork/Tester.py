from torch import nn

import src.Validation.DataLoader as DL
from src.Validation.Validation import get_accuracy_estimate_for_network
from src2.Configuration.Configuration import config
from src2.Genotype.CDN.Genomes.BlueprintGenome import BlueprintGenome
from src2.Genotype.CDN.Genomes.ModuleGenome import ModuleGenome

from src2.Genotype.NEAT.Connection import Connection
from src2.Genotype.NEAT.Node import Node, NodeType

from src2.Genotype.CDN.Nodes.ModuleNode import ModuleNode
from src2.Genotype.CDN.Nodes.BlueprintNode import BlueprintNode
from src2.Genotype.NEAT.Population import Population

from src2.Genotype.NEAT.Species import Species
from src2.Phenotype.NeuralNetwork.NeuralNetwork import Network
from src2.main.Generation import Generation

conn0 = Connection(0, 0, 2)
conn1 = Connection(1, 0, 3)
conn2 = Connection(2, 2, 5)
conn3 = Connection(3, 3, 6)
conn4 = Connection(4, 6, 4)
conn5 = Connection(5, 5, 1)
conn6 = Connection(6, 4, 1)
conn7 = Connection(7, 3, 5)

# conn3.enabled.set_value(False)

n0 = ModuleNode(0, NodeType.INPUT)
n1 = ModuleNode(1, NodeType.OUTPUT)
n2 = ModuleNode(2)
n3 = ModuleNode(3)
n4 = ModuleNode(4)
n5 = ModuleNode(5)
n6 = ModuleNode(6)
genome0 = ModuleGenome([n0, n1, n2, n3, n4, n5, n6], [conn0, conn1, conn2, conn3, conn4, conn5, conn6, conn7])

conn10 = Connection(0, 0, 1)
n10 = ModuleNode(0, NodeType.INPUT)
n11 = ModuleNode(1, NodeType.OUTPUT)

# n10.layer_type.set_value(nn.Linear)
genome1 = ModuleGenome([n10, n11], [conn10])

spcs = [Species(genome0), Species(genome1)]

bn0 = BlueprintNode(0, NodeType.INPUT)
bn1 = BlueprintNode(1, NodeType.OUTPUT)

bn0.species_id = 0
bn1.species_id = 1

bpg = BlueprintGenome([bn0, bn1], [conn10])

x, target = DL.sample_data(config.get_device(), 2)
enen = Network(bpg, spcs, list(x.shape)).to(config.get_device())
li = enen.model.get_layer_info()
print(li)
enen.visualize()

# get_accuracy_estimate_for_network(enen)
