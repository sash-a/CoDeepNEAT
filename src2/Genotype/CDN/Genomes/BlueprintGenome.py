from typing import List

from src2.Genotype.Mutagen.ContinuousVariable import ContinuousVariable
from src2.Genotype.Mutagen.Mutagen import Mutagen
from src2.Genotype.NEAT.Connection import Connection
from src2.Genotype.NEAT.Genome import Genome
from src2.Genotype.NEAT.Node import Node


class BlueprintGenome(Genome):

    def __init__(self, nodes: List[Node], connections: List[Connection]):
        super().__init__(nodes,connections)

        self.learning_rate = ContinuousVariable("learning rate",  start_range=0.0006, current_value=0.001, end_range=0.003, mutation_chance= 0)
        self.beta1 = ContinuousVariable("beta1",  start_range=0.88, current_value=0.9, end_range=0.92, mutation_chance= 0)
        self.beta2 = ContinuousVariable("beta2",  start_range=0.9988, current_value=0.999, end_range=0.9992, mutation_chance= 0)

    def get_modules_used(self):
        """:returns all module ids currently being used by this blueprint. returns duplicates"""
        pass

    def get_all_mutagens(self) -> List[Mutagen]:
        return [self.learning_rate, self.beta1, self.beta2]

