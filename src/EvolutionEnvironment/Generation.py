from src.Module.Species import Species
from src.Blueprint.Blueprint import BlueprintNode
from src.Graph import Node
from src.NeuralNetwork.ANN import ModuleNet
import torch
import math

class Generation:

    numBlueprints = 1
    numModules = 1

    speciesCollection = {} #hashmap from species number to species
    speciesNumbers = []
    blueprintCollection = set()

    def __init__(self, firstGen = False, previousGeneration = None):
        self.speciesNumbers = []
        self.speciesCollection = {}
        if(firstGen):
            self.initialisePopulation()
        else:
            self.generateFromPreviousGeneration(previousGeneration)

    def initialisePopulation(self):
        print("initialising random population")

        for b in range(self.numBlueprints):
            blueprint = Node.genNodeGraph(BlueprintNode, "linear")
            self.blueprintCollection.add(blueprint)

        species = Species()
        species.initialiseModules(self.numModules)
        self.speciesCollection[species.speciesNumber] = species
        self.speciesNumbers.append(species.speciesNumber)


    def generateFromPreviousGeneration(self, previousGen):
        pass

    def evaluate(self):
        print("evaluating blueprint")

        for blueprint in self.blueprintCollection:
            print("parsing blueprint to module")
            moduleGraph = blueprint.parsetoModule(self)
            #moduleGraph.printTree()
            moduleGraph.insertAggregatorNodes()
            #print(moduleGraph.getOutputNode().traversalID)
            moduleGraph.plotTree(set(),math.radians(0))

            net = ModuleNet(moduleGraph)

            out = net(torch.randn(1,5))

            print("final output:",out)






