import src.Config.NeatProperties as Props
from src.NEAT.MultiobjectivePopulation import MultiobjectivePopulation

from src.NeuralNetwork import Evaluator
from src.CoDeepNEAT import PopulationInitialiser as PopInit
from src.Analysis import RuntimeAnalysis
from src.Config import Config

module_population = None
import random


class Generation:
    numBlueprints = 1
    numModules = 5

    def __init__(self):
        self.speciesNumbers = []
        self.module_population, self.blueprint_population = self.initialise_populations()

    def initialise_populations(self):
        print('initialising population')
        global module_population
        module_population = MultiobjectivePopulation(PopInit.initialise_modules(),
                                                     PopInit.initialize_mutations(),
                                                     Props.MODULE_POP_SIZE,
                                                     Props.MODULE_NODE_MUTATION_CHANCE,
                                                     Props.MODULE_CONN_MUTATION_CHANCE,
                                                     Props.MODULE_TARGET_NUM_SPECIES)

        blueprint_population = MultiobjectivePopulation(PopInit.initialise_blueprints(),
                                                        PopInit.initialize_mutations(),
                                                        Props.BP_POP_SIZE,
                                                        Props.BP_NODE_MUTATION_CHANCE,
                                                        Props.BP_CONN_MUTATION_CHANCE,
                                                        Props.BP_TARGET_NUM_SPECIES)

        print('population initialized')
        return module_population, blueprint_population

    def step(self):
        """Runs CDN for one generation - must be called after fitness evaluation"""

        self.module_population.step()
        for blueprint_individual in self.blueprint_population.individuals:
            blueprint_individual.reset_number_of_module_species(module_population.get_num_species())
        self.blueprint_population.step()  # TODO should blueprints be speciatied ?

        for blueprint_individual in self.blueprint_population.individuals:
            blueprint_individual.clear()

        for module_individual in self.module_population.individuals:
            module_individual.clear()  # this also sets fitness to zero

    def evaluate(self, generation_number):
        print("num species:", len(self.module_population.species))
        inputs, targets = Evaluator.sample_data('mnist', '../../data')

        best_acc = -9999999999999999999
        best_bp = None
        accuracies = []
        second_objective_values = []
        best_second = -9999999999999999999
        third_objective_values = []
        best_third = -9999999999999999999


        # Randomize the list so that random individuals are sampled more often
        random.shuffle(self.blueprint_population.individuals)

        for i, blueprint_individual in enumerate(self.blueprint_population.individuals):
            # Checking if done enough evals
            if Props.INDIVIDUALS_TO_EVAL < i:
                break

            if Config.protect_parsing_from_errors:
                try:
                    module_graph, blueprint_individual, results = self.evaluate_blueprints(blueprint_individual, inputs, generation_number)
                except Exception as e:
                    blueprint_individual.defective = True
                    print(e)
                    print("blueprint indv ran with errors")
                    continue
            else:
                module_graph, blueprint_individual, results = self.evaluate_blueprints(blueprint_individual, inputs,generation_number)

            second = third = None
            if len(results) == 1:
                acc = results
            elif len(results) == 2:
                acc, second = results
            elif len(results) == 3:
                acc, second, third = results
            else:
                raise Exception("Error: too many result values to unpack")

            if acc >= best_acc:
                best_acc = acc
                best_bp = module_graph
            accuracies.append(acc)

            if not (second is None):
                best_second = max(best_second, second)
                second_objective_values.append(second)
            if not (third is None):
                best_third = max(best_third, third)
                third_objective_values.append(third)


        if generation_number % Config.print_best_graph_every_n_generations == 0:
            if Config.print_best_graphs:
                best_bp.plot_tree(title="gen:" + str(generation_number) + " acc:" + str(best_acc))

        RuntimeAnalysis.log_new_generation(accuracies, generation_number,
                                           second_objective_values= (second_objective_values if len(second_objective_values)>0 else None),
                                           third_objective_values = (third_objective_values if len(third_objective_values)>0 else None))
        print('best acc', best_acc)

    def evaluate_blueprints(self, blueprint_individual, inputs, generation_number):

        blueprint = blueprint_individual.to_blueprint()
        module_graph = blueprint.parseto_module_graph(self)
        if module_graph is None:
            raise Exception("null module graph produced from blueprint")

        # net = module_graph.to_nn(in_features=1, device=device)
        try:
            # print("using infeatures = ",module_graph.get_first_feature_count(inputs))
            net = module_graph.to_nn(in_features=module_graph.get_first_feature_count(inputs))
        except Exception as e:
            print("Error:", e)
            if Config.print_failed_graphs:
                module_graph.plot_tree_with_matplotlib("module graph which failed to parse to nn")
            raise Exception("Error: failed to parse module graph into nn")

        try:
            net.specify_dimensionality(inputs)
        except Exception as e:
            print("Error:", e)
            if Config.print_failed_graphs:
                module_graph.plot_tree_with_matplotlib(title="module graph with error passing input through net")
            raise Exception("Error: nn failed to have input passed through")

        if Config.dummy_run and generation_number < 500:
            acc = hash(net)
        else:
            acc = Evaluator.evaluate(net, 2, dataset='mnist', path='../../data', batch_size=256)

        second_objective_value = None
        third_objective_value = None

        if Config.second_objective == "network_size":
            second_objective_value = net.module_graph.get_net_size()
        elif Config.second_objective == "":
            pass
        else:
            print("Error: did not recognise second objective",Config.second_objective)

        if second_objective_value is None:
            results = acc
        elif third_objective_value is None:
            results = acc, second_objective_value
        else:
            results = acc, second_objective_value, third_objective_value

        blueprint_individual.report_fitness(*results)
        for module_individual in blueprint_individual.modules_used:
            module_individual.report_fitness(*results)

        return module_graph, blueprint_individual, results
