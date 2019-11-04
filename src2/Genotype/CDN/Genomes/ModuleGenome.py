from typing import List

from src2.Genotype.NEAT.Connection import Connection
from src2.Genotype.NEAT.Genome import Genome
from src2.Genotype.NEAT.Node import Node


class ModuleGenome(Genome):

    def __int__(self, nodes: List[Node], connections: List[Connection]):
        super().__init__(nodes, connections)

    def to_phenotype(self, bp_node_id: int):
        pass
