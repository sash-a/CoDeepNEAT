from src.Phenotype.BlueprintNode import BlueprintNode


class BlueprintGraph():

    def __init__(self, root_node):
        self.root_node: BlueprintNode = root_node

    def parse_to_module_graph(self, generation):
        return self.root_node.parse_to_module_graph(generation)
