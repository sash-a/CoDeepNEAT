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
        self.blueprint_population.step()  # TODO should blueprints be speciatied ?
        self.module_population.step()

        for blueprint_individual in self.blueprint_population.individuals:
            blueprint_individual.clear(module_population.get_num_species())

        for module_individual in self.module_population.individuals:
            module_individual.clear()  # this also sets fitness to zero

    def evaluate(self, generation_number):
        print("num species:", len(self.module_population.species))
        inputs, targets = Evaluator.sample_data('mnist', '../../data')

        best_acc = -9999999999999999999
        best_bp = None
        accuracies = []

        # Randomize the list so that random individuals are sampled more often
        random.shuffle(self.blueprint_population.individuals)

        for i, blueprint_individual in enumerate(self.blueprint_population.individuals):
            # Checking if done enough evals
            if Props.INDIVIDUALS_TO_EVAL < i:
                break

            if Config.protect_parsing_from_errors:
                try:
                    acc, module_graph = self.evaluate_blueprints(blueprint_individual , inputs,generation_number)
                except:
                    print("blueprint indv ran with errors")
                    continue
            else:
                acc, module_graph, blueprint_individual = self.evaluate_blueprints(blueprint_individual, inputs, generation_number)

            if acc >= best_acc:
                best_acc = acc
                best_bp = module_graph
            accuracies.append(acc)

        if(generation_number % 1 == 0):
            if(Config.print_graphs):
                best_bp.plot_tree(title="gen:"+str(generation_number)+" acc:"+str(best_acc))

        RuntimeAnalysis.log_new_generation(accuracies,generation_number)
        print('\n\nbest acc', best_acc)

    def evaluate_blueprints(self, blueprint_individual, inputs, generation_number):

        blueprint = blueprint_individual.to_blueprint()
        module_graph = blueprint.parseto_module_graph(self)
        if(module_graph is None):
            raise Exception("null module graph produced from blueprint")

        # net = module_graph.to_nn(in_features=1, device=device)
        try:
            #print("using infeatures = ",module_graph.get_first_feature_count(inputs))
            net = module_graph.to_nn(in_features=module_graph.get_first_feature_count(inputs))
        except Exception as e:
            print("Error:", e)
            module_graph.plot_tree("module graph which failed to parse to nn")
            raise Exception("Error: failed to parse module graph into nn")

        try:
            net.specify_dimensionality(inputs)
        except Exception as e:
            print("Error:", e)
            module_graph.plot_tree(title="module graph with error passing input through net")
            raise Exception("Error: nn failed to have input passed through")

        if Config.dummy_run and generation_number < 500:
            acc = hash(net)
        else:
            acc = Evaluator.evaluate(net, 2, dataset='mnist', path='../../data', batch_size=256)

        net_size = net.module_graph.get_net_size()

        blueprint_individual.report_fitness(acc,net_size)

        for module_individual in blueprint_individual.modules_used:
            module_individual.report_fitness(acc,net_size)



        return acc, module_graph, blueprint_individual
