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
        speciesCounter += 1

    def sample_module(self):
        index = randint(0, len(self.moduleCollection) - 1)
        return copy.deepcopy(self.moduleCollection[index]), index

    def initialise_modules(self, num_modules):
        print("initialising modules")

        for m in range(num_modules):
            module = Node.gen_node_graph(ModuleNode, "diamond")
            self.moduleCollection.append(module)
