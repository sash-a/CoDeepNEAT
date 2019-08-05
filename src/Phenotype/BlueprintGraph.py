from src.Phenotype.BlueprintNode import BlueprintNode
from src.Config import Config

class BlueprintGraph():

    def __init__(self, root_node):
        self.root_node: BlueprintNode = root_node

    def parse_to_module_graph(self, generation):
        if (generation.generation_number == 0 or not Config.module_retention) \
                and self.root_node.blueprint_genome.species_module_index_map:
            raise Exception("expected empty species index map, got" +
                            repr(self.root_node.blueprint_genome.species_module_index_map))
        return self.root_node.parse_to_module_graph(generation)
