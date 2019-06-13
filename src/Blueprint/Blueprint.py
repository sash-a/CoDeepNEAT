from src.Graph.Node import Node


class BlueprintNode(Node):

    """
    Each value in a blueprint graph is a Module Species number
    """

    speciesIndexesUsed = []

    def __init__(self):
        Node.__init__(self)
        self.value = 0
        self.speciesIndexesUsed = []

    def parsetoModule(self, generation ,moduleConstruct = None, speciesindexes = None):
        """
        :param moduleConstruct: the output module node to have this newly sampled module attached to. None if this is root blueprint node
        :return:
        """
        inputModuleNode, index = generation.speciesCollection[self.value].sampleModule()#to be added as child to existing module construct
        outputModuleNode = inputModuleNode.getOutputNode()#many branching modules may be added to this module


        if(not moduleConstruct == None):
            moduleConstruct.addChild(inputModuleNode)
        else:
            if(not self.isInputNode()):
                print("null module construct passed to non root blueprint node")

        if(self.isInputNode()):
            self.speciesIndexesUsed = []
            speciesindexes = self.speciesIndexesUsed
            speciesindexes.append(index)


        for childBlueprintNode in self.children:
            childBlueprintNode.parsetoModule(generation, outputModuleNode, speciesindexes)#passes species index down to collect all species indexes used to construct this blueprint in one list

        if (len(self.parents) == 0):
            print("blueprint parsed. getting module node traversal ID's")
            inputModuleNode.getTraversalIDs("_")
            return inputModuleNode