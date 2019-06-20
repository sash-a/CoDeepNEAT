from src.Graph.Node import Node
from src.Module.ModuleNode import ModuleNode
from torch import nn

class BlueprintNode(Node):

    """
    Each value in a blueprint graph is a Module Species number
    """

    speciesIndexesUsed = []

    def __init__(self):
        Node.__init__(self)
        self.value = 0
        self.speciesIndexesUsed = []

    def parseto_module(self, generation, moduleConstruct=None, speciesindexes=None):
        """
        :param moduleConstruct: the output module node to have this newly sampled module attached to. None if this is root blueprint node
        :return:
        """
        inputModuleNode, index = generation.speciesCollection[
            self.value].sample_module()  # to be added as child to existing module construct
        outputModuleNode = inputModuleNode.get_output_node()  # many branching modules may be added to this module

        if (not moduleConstruct == None):
            moduleConstruct.add_child(inputModuleNode)
        else:
            if (not self.is_input_node()):
                print("null module construct passed to non root blueprint node")

        if (self.is_input_node()):
            self.speciesIndexesUsed = []
            speciesindexes = self.speciesIndexesUsed
            speciesindexes.append(index)


        for childBlueprintNode in self.children:
            childBlueprintNode.parseto_module(generation, outputModuleNode,
                                              speciesindexes)  # passes species index down to collect all species indexes used to construct this blueprint in one list

        if (len(self.parents) == 0):
            # print("blueprint parsed. getting module node traversal ID's")
            inputModuleNode.get_traversal_ids("_")
            return inputModuleNode
