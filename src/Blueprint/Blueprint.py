from src.Graph.Node import Node
from src.Module.ModuleNode import ModuleNode
from torch import nn

class BlueprintNode(Node):

    """
    Each value in a blueprint graph is a Module Species number
    """

    speciesIndexesUsed = []

    def __init__(self, blueprint_genome):
        Node.__init__(self)
        self.value = 0#which species to sample from
        self.speciesIndexesUsed = []
        self.module_root = None#when this blueprint node is parsed to a module - this will be a ref to the input node of that module
        self.module_leaf = None#upon parsing this will hold the output node of the module\
        self.blueprint_genome = blueprint_genome

    def parseto_module_graph(self, generation, moduleConstruct=None, speciesindexes=None):
        """
        :param moduleConstruct: the output module node to have this newly sampled module attached to. None if this is root blueprint node
        :return: a handle on the root node of the newly created module graph
        """

        if(self.module_root is None and self.module_leaf is None):
            #first time this blueprint node has been reached in the traversal
            inputModuleIndividual, index = generation.module_population.species[self.value].sample_individual()  # to be added as child to existing module construct
            self.blueprint_genome.modules_used.append(inputModuleIndividual)
            inputModuleNode = inputModuleIndividual.to_module()

            outputModuleNode = inputModuleNode.get_output_node()  # many branching modules may be added to this module
            self.module_leaf = outputModuleNode
            self.module_root = inputModuleNode
            first_traversal = True
        else:
            inputModuleNode = self.module_root
            outputModuleNode = self.module_leaf
            first_traversal = False


        if (not moduleConstruct == None):
            moduleConstruct.add_child(inputModuleNode)
        else:
            if (not self.is_input_node()):
                print("null module construct passed to non root blueprint node")

        # if (self.is_input_node()):
        #     self.speciesIndexesUsed = []
        #     speciesindexes = self.speciesIndexesUsed
        #     speciesindexes.append(index)


        if(first_traversal):
            for childBlueprintNode in self.children:
                childBlueprintNode.parseto_module_graph(generation, outputModuleNode, speciesindexes)  # passes species index down to collect all species indexes used to construct this blueprint in one list

        if (len(self.parents) == 0):
            # print("blueprint parsed. getting module node traversal ID's")
            inputModuleNode.get_traversal_ids("_")
            return inputModuleNode
