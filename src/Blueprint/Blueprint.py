from CoDeepNEAT.src.Graph.Node import Node
from CoDeepNEAT.src.Module.ModuleNode import ModuleNode
from CoDeepNEAT.src.EvolutionEnvironment import EvolutionEnvironment

class BlueprintNode(Node):

    """
    Each value in a blueprint graph is a Module Species number
    """

    speciesIndexesUsed = []

    def __init__(self, level = 0):
        Node.__init__(self,None,0)
        pass

    def parsetoModule(self, moduleConstruct = None, speciesindexes = None):
        """

        :param moduleConstruct: the output module node to have this newly sampled module attached to. None if this is root blueprint node
        :return:
        """

        inputModuleNode, index = EvolutionEnvironment.currentGeneration.speciesCollection[self.value].sampleModule()#to be added as child to existing module construct
        outputModuleNode = inputModuleNode.getLeafNode()#many branching modules may be added to this module

        if(not moduleConstruct == None):
            moduleConstruct.addChild(inputModuleNode)
            if(self.level == 0):
                self.speciesIndexesUsed = []
                speciesindexes = self.speciesIndexesUsed
                speciesindexes.append(index)
            else:
                print("null module construct passed to none input blueprint node")
                return None

        for childBlueprintNode in self.children:
            childBlueprintNode.parsetoModule(outputModuleNode, speciesindexes)#passes species index down to collect all species indexes used to construct this blueprint in one list

        if(self.level == 0):
            inputModuleNode.getTraversalIDs()
            return inputModuleNode