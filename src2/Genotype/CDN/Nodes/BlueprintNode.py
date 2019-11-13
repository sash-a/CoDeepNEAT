from __future__ import annotations

import random
from typing import Dict, TYPE_CHECKING, Tuple

from src2.Genotype.Mutagen.IntegerVariable import IntegerVariable
from src2.Genotype.NEAT.Node import Node, NodeType
from src2.Phenotype.NeuralNetwork.Layers.Layer import Layer

if TYPE_CHECKING:
    from src2.Genotype.CDN.Genomes.ModuleGenome import ModuleGenome


class BlueprintNode(Node):
    def __init__(self, id: int, type: NodeType):
        super().__init__(id, type)

        import src2.main.Singleton as Singleton

        self.linked_module_id: int = -1
        self.module_repeat_count = IntegerVariable("module_repeat_count", start_range=1, current_value=1, end_range=4,
                                                   mutation_chance=0.1)

        # TODO pick again when more species are created
        possible_species_ids = [spc.id for spc in Singleton.instance.module_population.species]
        self.species_id: int = random.choice(possible_species_ids)

    def get_all_mutagens(self):
        return [self.module_repeat_count]

    def pick_module(self, module_sample_map: Dict[int, int]) -> ModuleGenome:
        # import src2.main.Generation as Generation  # TODO how slow is this
        import src2.main.Singleton as S

        if self.linked_module_id != -1:
            return S.instance.module_population[self.linked_module_id]

        if self.species_id not in module_sample_map:
            species = S.instance.module_population.get_species_by_id(self.species_id)
            if species is None:
                raise Exception('Species with ID = ' + str(self.species_id) + ' no longer exists')

            module_sample_map[self.species_id] = species.sample_individual().id

        self.linked_module_id = module_sample_map[self.species_id]
        return S.instance.module_population[module_sample_map[self.species_id]]

    def convert_node(self, **kwargs) -> Tuple[Layer, Layer]:
        # TODO module sampling needs to live long enough to be able to be committed
        module_sample_map = kwargs['module_sample_map']
        return self.pick_module(module_sample_map).to_phenotype(blueprint_node_id=self.id)
