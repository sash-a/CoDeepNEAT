from src.NEAT.Population import Population
from src.NeuralNetwork import Evaluator
from src.CoDeepNEAT import PopulationInitialiser
import torch.tensor


class Generation:
    numBlueprints = 1
    numModules = 5

    def __init__(self):
        self.speciesNumbers = []
        self.module_population, self.blueprint_population = self.initialise_populations()

    def initialise_populations(self):
        print("initialising random population")

        module_population = Population(PopulationInitialiser.initialise_modules())
        blueprint_population = Population(PopulationInitialiser.initialise_blueprints())

        return module_population, blueprint_population

    def step(self):
        """Runs CDN for one generation - must be called after fitness evaluation"""
        self.blueprint_population.step()  # TODO should blueprints be speciatied ?
        self.module_population.step()

    def evaluate(self, device=torch.device("cuda:0"), print_graphs=False):
        inputs, targets = Evaluator.sample_data('mnist', '../../data', device=device)

        for blueprint_individual in self.blueprint_population.individuals:

            blueprint = blueprint_individual.to_blueprint()
            module_graph = blueprint.parseto_module_graph(self)
            net = module_graph.toNN(in_features=1, device=device)

            net.specify_output_dimensionality(inputs, device=device)

            acc = Evaluator.evaluate(net, 15, dataset='mnist', path='../../data', device=device, batch_size=256)
            blueprint_individual.report_fitness(acc)

            for module_individual in blueprint_individual.modules_used:
                module_individual.report_fitness()

            blueprint_individual.clear()

        for module_individual in self.module_population.individuals:
            module_individual.clear()
