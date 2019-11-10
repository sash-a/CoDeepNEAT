from typing import List, Tuple

from src2.Phenotype.NeuralNetwork.Layers.Layer import Layer
from src2.Genotype.NEAT.Connection import Connection
from src2.Genotype.NEAT.Genome import Genome
from src2.Genotype.NEAT.Node import Node


class ModuleGenome(Genome):

    def __int__(self, nodes: List[Node], connections: List[Connection]):
        super().__init__(nodes, connections)

    def to_phenotype(self, **kwargs) -> Tuple[Layer, Layer]:
        return super().to_phenotype(**kwargs)
