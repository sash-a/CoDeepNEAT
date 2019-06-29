from src.Graph.Node import Node
from src.Module.ModuleNode import ModuleNode
from torch import nn
import torch


class BlueprintNode(Node):
    """
    Each value in a blueprint graph is a Module Species number
    """

    speciesIndexesUsed = []

    def __init__(self, blueprint_NEAT_node, blueprint_genome):
        Node.__init__(self)
        self.species_number = 0  # which species to sample from
        self.speciesIndexesUsed = []
        self.module_root = None  # when this blueprint node is parsed to a module - this is a ref to input node
        self.module_leaf = None  # upon parsing this will hold the output node of the module
        self.blueprint_genome = blueprint_genome

        if blueprint_NEAT_node is None:
            print("null neat node passed to blueprint")
        self.generate_blueprint_node_from_gene(blueprint_NEAT_node)

    def generate_blueprint_node_from_gene(self, gene):
        """applies the properties of the blueprint gene for this node"""
        self.species_number = gene.species_number

    def parseto_module_graph(self, generation, module_construct=None, species_indexes=None, in_features=1,
                             device=torch.device("cpu")):
        """
        :param module_construct: the output module node to have this newly sampled module attached to. None if this is root blueprint node
        :return: a handle on the root node of the newly created module graph
        """

        if self.module_root is None and self.module_leaf is None:
            # first time this blueprint node has been reached in the traversal
            # to be added as child to existing module construct
            input_module_individual, index = generation.module_population.species[
                self.species_number].sample_individual()
            self.blueprint_genome.modules_used.append(input_module_individual)
            input_module_node = input_module_individual.to_module_node()
            if (not input_module_node.is_input_node()):
                print("error! sampled module node is not root node")

            # many branching modules may be added to this module
            output_module_node = input_module_node.get_output_node()
            self.module_leaf = output_module_node
            self.module_root = input_module_node
            first_traversal = True
        else:
            input_module_node = self.module_root
            output_module_node = self.module_leaf
            first_traversal = False

        if module_construct is not None:
            module_construct.add_child(input_module_node)
        else:
            if not self.is_input_node():
                print("null module construct passed to non root blueprint node")

        if first_traversal:
            # passes species index down to collect all species indexes used to construct this blueprint in one list
            for childBlueprintNode in self.children:
                childBlueprintNode.parseto_module_graph(generation, output_module_node, species_indexes)

        if len(self.parents) == 0:
            # print("blueprint parsed. getting module node traversal ID's")
            input_module_node.get_traversal_ids("_")
            try:
                input_module_node.insert_aggregator_nodes()
            except:
                input_module_node.plot_tree()
                print("failed to inser")
            input_module_node.create_layers(in_features=in_features)


            return input_module_node
