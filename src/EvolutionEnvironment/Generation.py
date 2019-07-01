from src.NEAT.Population import Population
from src.NeuralNetwork import Evaluator
from src.CoDeepNEAT import PopulationInitialiser
import torch.tensor

import random


class Generation:
    numBlueprints = 1
    numModules = 5

    def __init__(self):
        self.speciesNumbers = []
        self.module_population, self.blueprint_population = self.initialise_populations()

    def initialise_populations(self):
        print('initialising population')
        module_population = Population(PopulationInitialiser.initialise_modules())
        blueprint_population = Population(PopulationInitialiser.initialise_blueprints())
        print('population initialized')
        return module_population, blueprint_population

    def step(self):
        """Runs CDN for one generation - must be called after fitness evaluation"""
        self.blueprint_population.step()  # TODO should blueprints be speciatied ?
        self.module_population.step()

        for blueprint_individual in self.blueprint_population.individuals:
            blueprint_individual.clear()

        for module_individual in self.module_population.individuals:
            module_individual.clear()  # this also sets fitness to zero

    def evaluate(self, generation_number, device=torch.device("cuda:0"), print_graphs=True):
        inputs, targets = Evaluator.sample_data('mnist', '../../data', device=device)

        for blueprint_individual in self.blueprint_population.individuals:
            print('\n\nTraining next blueprint')
            blueprint = blueprint_individual.to_blueprint()
            if not blueprint.is_input_node():
                print("Error: blueprint graph handle node is not root node")

            module_graph = blueprint.parseto_module_graph(self, device=device)
            if module_graph is None:
                print("failed parsing blueprint to module graph")
                continue

            if print_graphs and random.random() > 0.8:
                print('\n\nPRINTING BLUEPRINTS AND MODULES\n')
                # blueprint.plot_tree(title="blueprint")
                module_graph.plot_tree(title="module graph")
                print('Blueprint')
                print(blueprint_individual)
                print('modules')
                for module in blueprint_individual.modules_used:
                    print(module)

            if not module_graph.is_input_node():
                print("Error: module graph handle node is not root node")

            net = module_graph.to_nn(in_features=1, device=device)
            # try:
            #     net = module_graph.to_nn(in_features=1, device=device)
            # except Exception as e:
            #     print("Error:",e)
            #     print("Error: failed to parse module graph into nn")
            #     module_graph.plot_tree()
            #     continue

            try:
                net.specify_output_dimensionality(inputs, device=device)
                # if (generation > 0):
                #     module_graph.plot_tree(title="module graph without error")

            except Exception as e:
                print("Error:", e)
                print("Error: nn failed to have input passed through")
                module_graph.plot_tree(title="module graph with error")
                continue

            acc = hash(net)  # Evaluator.evaluate(net, 1, dataset='mnist', path='../../data', device=device, batch_size=64)
            blueprint_individual.report_fitness(acc)

            for module_individual in blueprint_individual.modules_used:
                module_individual.report_fitness(acc)
