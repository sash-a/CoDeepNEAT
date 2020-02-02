from __future__ import annotations

import random
from typing import Dict, TYPE_CHECKING, Tuple, List

# For testing!
from src2.configuration import config
from src2.genotype.mutagen.integer_variable import IntegerVariable
from src2.genotype.neat.node import Node, NodeType
from src2.phenotype.neural_network.layers.layer import Layer

if TYPE_CHECKING:
    from src2.genotype.cdn.genomes.module_genome import ModuleGenome


class BlueprintNode(Node):

    def __init__(self, id: int, type: NodeType):
        super().__init__(id, type)

        import src2.main.singleton as Singleton

        self.linked_module_id: int = -1
        self.module_repeats = IntegerVariable("module_repeat_count", start_range=1, current_value=1,
                                              end_range=config.max_module_repeats,
                                              mutation_chance=0.1 if config.max_module_repeats > 1 else 0)

        # TODO pick again when more species are created
        possible_species_ids = [spc.id for spc in Singleton.instance.module_population.species]
        self.species_id: int = random.choice(possible_species_ids)

    def get_all_mutagens(self):
        node_muts = super().get_all_mutagens()
        node_muts.extend([self.module_repeats])
        return node_muts

    def pick_module(self, module_sample_map: Dict[int, int], ignore_species: List[int]) -> ModuleGenome:
        import src2.main.singleton as Singleton

        if self.linked_module_id != -1 and self.species_id not in ignore_species:
            # genome already linked module
            module = Singleton.instance.module_population[self.linked_module_id]
            if module is None:
                raise Exception("bad module link " + str(self.linked_module_id))
            module_sample_map[self.species_id] = module.id
            return module

        # no genomic linked module

        if self.species_id not in module_sample_map:
            # unlinked module has not yet been sampled. Must sample module, add to sample map
            species = Singleton.instance.module_population.get_species_by_id(self.species_id)
            if species is None:
                raise Exception('Species with ID = ' + str(self.species_id) + ' no longer exists')

            module_sample_map[self.species_id] = species.sample_individual().id

        return Singleton.instance.module_population[module_sample_map[self.species_id]]

    def convert_node(self, **kwargs) -> Tuple[Layer, Layer]:
        module_sample_map = kwargs['module_sample_map']
        ignore_species = kwargs['ignore_species'] if 'ignore_species' in kwargs else -1
        module = self.pick_module(module_sample_map, ignore_species)
        if module is None:
            raise Exception("failed to sample module ")

        parent_output_layer: Layer
        first_input_layer, parent_output_layer = module.to_phenotype(blueprint_node_id=self.id)

        # Creating and linking this blueprints modules in a chain depending on how many repeats there are
        for i in range(self.module_repeats.value - 1):
            id_suffix = 0.1 * i
            input_layer, output_layer = module.to_phenotype(blueprint_node_id=self.id + id_suffix)
            parent_output_layer.add_child(str(module.id + id_suffix).replace('.', '-'), input_layer)
            parent_output_layer = output_layer

        return first_input_layer, parent_output_layer
