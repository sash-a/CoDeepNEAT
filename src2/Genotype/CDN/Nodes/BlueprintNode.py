from __future__ import annotations

import random
from typing import Dict, List, TYPE_CHECKING, Tuple

from src2.Phenotype.NeuralNetwork.Layers.Layer import Layer
from src2.main.Generation import Generation
from src2.Genotype.Mutagen.IntegerVariable import IntegerVariable
from src2.Genotype.NEAT.Node import Node, NodeType

if TYPE_CHECKING:
    from src2.Genotype.CDN.Genomes.ModuleGenome import ModuleGenome
    from src2.Genotype.NEAT.Species import Species
    from src2.Genotype.NEAT.Population import Population


class BlueprintNode(Node):

    def __init__(self, id: int, type: NodeType):
        super().__init__(id, type)

        self.linked_module_id: int = -1
        self.module_repeat_count = IntegerVariable("module_repeat_count", start_range=1, current_value=1, end_range=4,
                                                   mutation_chance=0.1)
        self.species_id: int = -1

    def get_all_mutagens(self):
        return [self.module_repeat_count]

    def pick_module(self, module_sample_map: Dict[int, int]) -> ModuleGenome:
        if self.linked_module_id != -1:
            return Generation.instance.module_population[self.linked_module_id]
        if self.species_id not in module_sample_map:
            species = Generation.instance.module_population.get_species_by_id(self.species_id)
            if species is None:
                raise Exception('Species with ID = ' + str(self.species_id) + ' no longer exists')

            module_sample_map[self.species_id] = species.sample_individual().id

        return Generation.instance.module_population[module_sample_map[self.species_id]]

    def convert_node(self, **kwargs) -> Tuple[Layer, Layer]:
        module_sample_map = kwargs['module_sample_map']  # TODO module sampling needs to live long enough to be able to be committed
        return self.pick_module(module_sample_map).to_phenotype()
