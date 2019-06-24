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
        #print("initialising modules")

        #initialisation_shapes = ["triangle", "diamond", "linear"]
        initialisation_shapes = ["triangle"]


        for m in range(num_modules ):

            shape = initialisation_shapes[randint(0, len(initialisation_shapes) - 1)]

            if(shape == "linear"):
                #module = Node.gen_node_graph(ModuleNode, shape, randint(1,3))
                module = Node.gen_node_graph(ModuleNode, shape, 5)

            else:
                module = Node.gen_node_graph(ModuleNode, shape)

            self.moduleCollection.append(module)
