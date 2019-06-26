from src.Module.Species import Species
from src.Blueprint.Blueprint import BlueprintNode
from src.Graph import Node

from src.NeuralNetwork import Evaluator
import torch.tensor


class Generation:

    numBlueprints = 1
    numModules = 5

    def __init__(self, first_gen=False, previous_generation=None):
        self.speciesNumbers = []
        self.module_species_collection = {} # hashmap from species number to species
        self.blueprintCollection = set()

        if first_gen:
            self.initialise_population()
        else:
            self.generate_from_previous_generation(previous_generation)

    def initialise_population(self):
        print("initialising random population")

        for b in range(self.numBlueprints):
            blueprint = Node.gen_node_graph(BlueprintNode, "triangle",linear_count = 1)
            self.blueprintCollection.add(blueprint)

        species = Species()
        species.initialise_modules(self.numModules)

        self.module_species_collection[species.speciesNumber] = species
        self.speciesNumbers.append(species.speciesNumber)

    def generate_from_previous_generation(self, previous_gen):
        print("generating new generation from previous")
        pass

    def evaluate(self, device=torch.device("cuda:0"), print_graphs = False):
        inputs, targets = Evaluator.sample_data('mnist', '../../data', device=device)

        # for individual in  pop
        #   blueprint = individual.to_blueprint()
        #   net = toNetwork(blueprint)
        #   acc = eval(net)
        #   individual.setFitness(acc)
        #   modules_used = individual.get_modules()
        #   for mod in modules:
        #       mod.increment_fitness()



        for blueprint in self.blueprintCollection:

            module_graph = blueprint.parseto_module(self)
            net = module_graph.toNN(in_features=1, device=device)
            net.specify_output_dimensionality(inputs, device=device)

            acc = Evaluator.evaluate(net, 15, dataset='mnist', path='../../data', device=device, batch_size= 256)


