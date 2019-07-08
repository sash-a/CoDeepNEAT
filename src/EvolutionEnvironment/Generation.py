from src.NEAT.Population import Population
from src.NEAT.MultiobjectivePopulation import MultiobjectivePopulation

from src.NeuralNetwork import Evaluator
from src.CoDeepNEAT import PopulationInitialiser as PopInit
import torch.tensor

from src.NeuralNetwork.Net import ModuleNet
import random


class Generation:
    numBlueprints = 1
    numModules = 5

    def __init__(self):
        self.speciesNumbers = []
        self.module_population, self.blueprint_population = self.initialise_populations()

    def initialise_populations(self):
        print('initialising population')
        module_population = MultiobjectivePopulation(PopInit.initialise_modules(), PopInit.initialize_mutations())
        blueprint_population = MultiobjectivePopulation(PopInit.initialise_blueprints(), PopInit.initialize_mutations())
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

        best_acc = 0
        best_bp = None
        best_size = 0

        for blueprint_individual in self.blueprint_population.individuals:
            blueprint = blueprint_individual.to_blueprint()
            if not blueprint.is_input_node():
                print("Error: blueprint graph handle node is not root node")

            module_graph = blueprint.parseto_module_graph(self, device=device)
            if module_graph is None:
                print("failed parsing blueprint to module graph")
                continue

            if print_graphs and random.random() > 0.8:
                print('\n\nPRINTING BLUEPRINTS AND MODULES\n')
                module_graph.plot_tree(title="module graph")
                print('NEAT Blueprint')
                print(blueprint_individual)
                print('NEAT Modules')
                for module in blueprint_individual.modules_used:
                    print(module)

            if not module_graph.is_input_node():
                print("Error: module graph handle node is not root node")

            try:
                net: ModuleNet = module_graph.to_nn(in_features=1, device=device)
            except Exception as e:
                print("Error:", e)
                print("Error: failed to parse module graph into nn")
                module_graph.plot_tree()
                continue

            try:
                net.specify_output_dimensionality(inputs, device=device)
            except Exception as e:
                print("Error: nn failed to have input passed through\n", e)
                module_graph.plot_tree(title="module graph with error")
                continue

            acc = 20  # Evaluator.evaluate(net, 2, dataset='mnist', path='../../data', device=device, batch_size=256)
            net_size = net.moduleGraph.get_net_size()

            if acc >= best_acc:
                best_acc = acc
                best_bp = module_graph
                best_size = net_size
            blueprint_individual.report_fitness(acc, net_size)

            for module_individual in blueprint_individual.modules_used:
                # TODO second objective should only be params of module
                module_individual.report_fitness(acc, net_size)

        print('best acc', best_acc, 'size', best_size)
        best_bp.plot_tree()
