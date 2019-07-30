from src.Phenotype.BlueprintNode import BlueprintNode

class BlueprintGraph():

    def __init__(self,blueprint_graph_root_node):
        self.blueprint_graph_root_node:BlueprintNode = blueprint_graph_root_node
        self.species_to_module_index_mapping = {}
        for node in blueprint_graph_root_node.get_all_nodes_via_bottom_up(set()):
            """give nodes a ref to their super graph"""
            node.blueprint_graph = self

    def parseto_module_graph(self, generation):
        module_graph, module_index_map = self.blueprint_graph_root_node.parseto_module_graph(generation)
        #print("got",module_index_map,"from parsing")
        self.species_to_module_index_mapping = module_index_map
        return module_graph

    def push_species_module_mapping_to_genome(self, accuracy):
        genome = self.blueprint_graph_root_node.blueprint_genome
        #print("clone pushing species module index mapping:", self.species_to_module_index_mapping, "to clone genome")
        genome.inherit_species_module_mapping_from_phenotype(self.species_to_module_index_mapping, accuracy)

