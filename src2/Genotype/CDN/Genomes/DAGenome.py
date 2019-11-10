
from __future__ import annotations

from typing import List, Dict, TYPE_CHECKING, Optional

from src2.Genotype.CDN.Nodes.DANode import DANode
from src2.Genotype.NEAT.Connection import Connection
from src2.Genotype.NEAT.Genome import Genome

from src2.Phenotype.Augmentations.AugmentationScheme import AugmentationScheme

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
