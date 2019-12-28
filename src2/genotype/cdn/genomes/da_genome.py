from __future__ import annotations

from typing import List

from src2.genotype.cdn.nodes.da_node import DANode
from src2.genotype.neat.connection import Connection
from src2.genotype.neat.genome import Genome
from src2.phenotype.augmentations.augmentation_scheme import AugmentationScheme


class DAGenome(Genome):

    def __init__(self, nodes: List[DANode], connections: List[Connection]):
        super().__init__(nodes, connections)

    def to_phenotype(self):
        """Construct a data augmentation scheme from its genome"""
        data_augmentations = []
        for node in self.nodes.values():
            data_augmentations.append(node)

        augmentation_scheme = AugmentationScheme(data_augmentations)
        return augmentation_scheme

    def validate(self) -> bool:
        traversal_dict = self.get_traversal_dictionary()
        for children in traversal_dict.values():
            if len(children) > 1:
                return False

        return super().validate()
