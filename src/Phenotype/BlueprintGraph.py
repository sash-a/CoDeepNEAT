from src.Phenotype.BlueprintNode import BlueprintNode

class BlueprintGraph():

    def __init__(self,blueprint_graph_root_node):
        self.blueprint_graph_root_node:BlueprintNode = blueprint_graph_root_node
        self.species_to_module_mapping = {}

    def parseto_module_graph(self, generation):
        module_graph, module_index_map = self.blueprint_graph_root_node.parseto_module_graph(generation)
        for species_used in module_index_map.keys():
            module_individual = generation.module_population.species[species_used][module_index_map[species_used]]
            self.species_to_module_mapping[species_used] = module_individual
        return module_graph

    def push_species_module_mapping_to_genome(self, accuracy):
        genome = self.blueprint_graph_root_node.blueprint_genome
        genome.inherit_species_module_mapping_from_phenotype(self.species_to_module_mapping, accuracy)