from __future__ import annotations

from typing import List, Union, TYPE_CHECKING

from src.configuration import config
from src.genotype.neat.connection import Connection
from src.genotype.neat.genome import Genome
from src.phenotype.augmentations.augmentation_scheme import AugmentationScheme
from src.phenotype.augmentations.batch_augmentation_scheme import BatchAugmentationScheme

if TYPE_CHECKING:
    from src.genotype.cdn.nodes.da_node import DANode


class DAGenome(Genome):
    def __init__(self, nodes: List[DANode], connections: List[Connection]):
        super().__init__(nodes, connections)

    def __repr__(self):
        da_nodes = ""
        for n_id in self.get_fully_connected_node_ids():
            node: DANode = self.nodes[n_id]
            kwargs = {}
            if node.da.value in node.da.submutagens:
                kwargs = {k: mutagen.value for k, mutagen in node.da.submutagens[node.da.value].items()}
            da_nodes += node.da.value + ": " + repr(kwargs) + "\n"
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
