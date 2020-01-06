from __future__ import annotations

from typing import List, Union

from src2.configuration import config
from src2.genotype.cdn.nodes.da_node import DANode
from src2.genotype.neat.connection import Connection
from src2.genotype.neat.genome import Genome
from src2.phenotype.augmentations.augmentation_scheme import AugmentationScheme
from src2.phenotype.augmentations.batch_augmentation_scheme import BatchAugmentationScheme


class DAGenome(Genome):
    def __init__(self, nodes: List[DANode], connections: List[Connection]):
        super().__init__(nodes, connections)

    def __repr__(self):
        da_nodes = ""
        for n in self.nodes.values():
            kwargs = {}
            if n.da.value in n.da.submutagens:
                kwargs = {k: mutagen.value for k, mutagen in n.da.submutagens[n.da.value].items()}
            da_nodes += n.da.value + ": " + repr(kwargs) + "\n"
        return da_nodes

    def to_phenotype(self) -> Union[BatchAugmentationScheme, AugmentationScheme]:
        """Construct a data augmentation scheme from its genome"""
        data_augmentations = []
        traversal = self.get_traversal_dictionary(exclude_disabled_connection=True)

        current_node_id = 0
        while current_node_id != 1:  # 1 is always the id of the output node
            aug = self.nodes[current_node_id].to_phenotype()
            data_augmentations.append(aug)

            if len(traversal[current_node_id]) != 1:
                raise Exception('DA node has branches')

            current_node_id = traversal[current_node_id][0]  # should always only be 1

        data_augmentations.append(self.nodes[1].to_phenotype())
        if config.batch_augmentation:
            return BatchAugmentationScheme(data_augmentations)
        else:
            return AugmentationScheme(data_augmentations)

    def validate(self) -> bool:
        traversal_dict = self.get_traversal_dictionary()
        for children in traversal_dict.values():
            if len(children) > 1:
                return False

        return super().validate()
