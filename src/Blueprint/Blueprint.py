from src.Graph.Node import Node
from src.Module.ModuleNode import ModuleNode
from torch import nn

class BlueprintNode(Node):

    """
    Each value in a blueprint graph is a Module Species number
    """

    speciesIndexesUsed = []

    def __init__(self, blueprint_NEAT_node , blueprint_genome ):
        Node.__init__(self)
        self.species_number = 0#which species to sample from
        self.speciesIndexesUsed = []
        self.module_root = None#when this blueprint node is parsed to a module - this will be a ref to the input node of that module
        self.module_leaf = None#upon parsing this will hold the output node of the module
        self.blueprint_genome = blueprint_genome

        self.generate_blueprint_node_from_gene(blueprint_NEAT_node)

    def generate_blueprint_node_from_gene(self, gene):
        """applies the properties of the blueprint gene for this node"""
        self.species_number = gene.species_number


    def parseto_module_graph(self, generation, moduleConstruct=None, speciesindexes=None):
        """
        :param moduleConstruct: the output module node to have this newly sampled module attached to. None if this is root blueprint node
        :return: a handle on the root node of the newly created module graph
        """

        if(self.module_root is None and self.module_leaf is None):
            #first time this blueprint node has been reached in the traversal
            input_module_individual, index = generation.module_population.species[self.species_number].sample_individual()  # to be added as child to existing module construct
            self.blueprint_genome.modules_used.append(input_module_individual)
            input_module_node = input_module_individual.to_module_node()

            output_module_node = input_module_node.get_output_node()  # many branching modules may be added to this module
            self.module_leaf = output_module_node
            self.module_root = input_module_node
            first_traversal = True
        else:
            input_module_node = self.module_root
            output_module_node = self.module_leaf
            first_traversal = False


        if (not moduleConstruct == None):
            moduleConstruct.add_child(input_module_node)
        else:
            if (not self.is_input_node()):
                print("null module construct passed to non root blueprint node")


        if(first_traversal):
            for childBlueprintNode in self.children:
                childBlueprintNode.parseto_module_graph(generation, output_module_node, speciesindexes)  # passes species index down to collect all species indexes used to construct this blueprint in one list

        if (len(self.parents) == 0):
            # print("blueprint parsed. getting module node traversal ID's")
            input_module_node.get_traversal_ids("_")
            return input_module_node
