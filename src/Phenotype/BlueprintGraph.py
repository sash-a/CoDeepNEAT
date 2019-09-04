import math
import random

from src.Config import Config
from src.Phenotype.BlueprintNode import BlueprintNode


class BlueprintGraph():

    def __init__(self, root_node):
        self.root_node: BlueprintNode = root_node

    def parse_to_module_graph(self, generation, allow_ignores):
        if Config.allow_species_module_mapping_ignores and Config.fitness_aggregation == "max" and allow_ignores:
            self.forget_mappings()

        if (generation.generation_number == 0 or not Config.module_retention) \
                and self.root_node.blueprint_genome.species_module_index_map:
            raise Exception("expected empty species index map, got" +
                            repr(self.root_node.blueprint_genome.species_module_index_map))
        return self.root_node.parse_to_module_graph(generation)

    def forget_mappings(self):
        nodes = self.root_node.get_all_nodes_via_bottom_up(set())
        species_numbers = set([node.species_number for node in nodes])
        number_of_species = len(species_numbers)

        maps = set(self.root_node.blueprint_genome.species_module_index_map.keys()).intersection(species_numbers)
        number_of_maps = len(maps)

        map_frac = number_of_maps / number_of_species
        # print("map frac:",map_frac,"p:", math.pow(map_frac, 1.5))

        if (1 > map_frac > 0 and random.random() < math.pow(map_frac, 1.5)) or map_frac == 1:
            ignore_species = random.choice(list(self.root_node.blueprint_genome.species_module_index_map.keys()))
            # print("ignoring spc", ignore_species)
            del self.root_node.blueprint_genome.species_module_index_map[ignore_species]
