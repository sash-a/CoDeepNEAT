from __future__ import annotations

from typing import List

from src2.genotype.cdn.nodes.da_node import DANode
from src2.genotype.neat.connection import Connection
from src2.genotype.neat.genome import Genome
from src2.phenotype.augmentations.augmentation_scheme import AugmentationScheme


class DAGenome(Genome):

    def __init__(self, nodes: List[DANode], connections: List[Connection]):
        super().__init__(nodes, connections)

    def __repr__(self):
        da_nodes = ""
        for n in self.nodes.values():
            kwargs = {k: mutagen.value for k, mutagen in n.da.submutagens[n.da.value].items()}
            da_nodes += n.da.value + ": " + repr(kwargs) + "\n"
        return da_nodes


    def to_phenotype(self):
        """Construct a data augmentation scheme from its genome"""
        data_augmentations = []
        node: DANode
        for node in self.nodes.values():
            data_augmentations.append(node.to_phenotype())

        augmentation_scheme = AugmentationScheme(data_augmentations)
        return augmentation_scheme

    def validate(self) -> bool:
        traversal_dict = self.get_traversal_dictionary()
        for children in traversal_dict.values():
            if len(children) > 1:
                return False

        return super().validate()
