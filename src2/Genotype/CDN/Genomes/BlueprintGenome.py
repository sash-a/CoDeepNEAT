from __future__ import annotations

from typing import List, Dict, TYPE_CHECKING, Optional, Set, Tuple

from torch import nn

from src2.Genotype.CDN.Nodes.BlueprintNode import BlueprintNode
from src2.Genotype.Mutagen.ContinuousVariable import ContinuousVariable
from src2.Genotype.Mutagen.Mutagen import Mutagen
from src2.Genotype.NEAT.Connection import Connection
from src2.Genotype.NEAT.Genome import Genome
from src2.Genotype.NEAT.Node import Node
from src2.Phenotype.NeuralNetwork.Layers.AggregationLayer import AggregationLayer

if TYPE_CHECKING:
    from src2.Genotype.NEAT.Species import Species
    from src2.Phenotype.NeuralNetwork.Layers import Layer
    from src2.Genotype.CDN.Genomes.ModuleGenome import ModuleGenome


class BlueprintGenome(Genome):

    def __init__(self, nodes: List[Node], connections: List[Connection]):
        super().__init__(nodes, connections)

        self.learning_rate = ContinuousVariable("learning rate", start_range=0.0006, current_value=0.001,
                                                end_range=0.003, mutation_chance=0)
        self.beta1 = ContinuousVariable("beta1", start_range=0.88, current_value=0.9, end_range=0.92, mutation_chance=0)
        self.beta2 = ContinuousVariable("beta2", start_range=0.9988, current_value=0.999, end_range=0.9992,
                                        mutation_chance=0)

        # mapping from species id to the genome id of the module sampled from that species
        self.best_module_sample_map: Optional[Dict[int, int]] = None  # todo empty this at the end of evaluation
        self.best_sample_map_fitness: float = 0

    def get_modules_used(self):
        """:returns all module ids currently being used by this blueprint. returns duplicates"""
        pass

    def get_all_mutagens(self) -> List[Mutagen]:
        return [self.learning_rate, self.beta1, self.beta2]

    def commit_sample_maps(self):
        """
            commits whatever species->module mapping is in the sample map
            this should be the best sampling found this step
        """
        for node in self.nodes.values():
            """node may be blueprint or module node"""
            if isinstance(node, BlueprintNode):
                """updates the module id value of each node in the genome according to the sample map present"""
                node.linked_module_id = self.best_module_sample_map[node.species_id]

    def to_phenotype(self, **kwargs) -> Tuple[Layer, Layer]:
        return super().to_phenotype(module_sample_map={})
