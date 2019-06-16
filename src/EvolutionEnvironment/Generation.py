from src.Module.Species import Species
from src.Blueprint.Blueprint import BlueprintNode
from src.Graph import Node
from src.NeuralNetwork.ANN import ModuleNet
import torch
import math

import torch.nn as nn
from src.Learner import Evaluator
import torch.tensor
from torch.utils.data import DataLoader

from torchvision import datasets, transforms


class Generation:
    numBlueprints = 1
    numModules = 1

    speciesCollection = {}  # hashmap from species number to species
    speciesNumbers = []
    blueprintCollection = set()

    def __init__(self, firstGen=False, previousGeneration=None):
        self.speciesNumbers = []
        self.speciesCollection = {}
        if (firstGen):
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
        print("evaluating blueprints")

        for blueprint in self.blueprintCollection:
            print("parsing blueprint to module")

            moduleGraph = blueprint.parsetoModule(self)
            moduleGraph.createLayers(inFeatures=1)
            moduleGraph.insertAggregatorNodes()
            #moduleGraph.plotTree(set(), math.radians(0))

            net = ModuleNet(moduleGraph)#.to(torch.device("cuda:0"))
            print("parsed blueprint to NN:",net)

            Evaluator.evaluate(net, 15, dataset='mnist', path='../data', device=torch.device("cpu") )


