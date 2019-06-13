from random import randint
from src.Graph import Node
from src.Module.ModuleNode import ModuleNode
import copy

speciesCounter = 0

class Species:

    """
    species contains a list of module input nodes
    """

    moduleCollection = []
    speciesNumber = -1

    def __init__(self):
        global speciesCounter
        self.moduleCollection = []
        self.speciesNumber = speciesCounter
        speciesCounter +=1
        

    def sampleModule(self):
        index = randint(0,len(self.moduleCollection)-1)
        #print(len(self.moduleCollection), index)
        return copy.deepcopy(self.moduleCollection[index]), index
    
    def initialiseModules(self, numModules):
        print("initialising modules")
        
        for m in range(numModules):
            module = Node.genNodeGraph(ModuleNode, "diamond")
            self.moduleCollection.append(module)