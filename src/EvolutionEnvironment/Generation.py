import src.Config.NeatProperties as Props
from src.NEAT.Population import Population, single_objective_rank, cdn_rank
from src.NeuralNetwork import Evaluator
from src.CoDeepNEAT import PopulationInitialiser as PopInit
from src.Analysis import RuntimeAnalysis
from src.Config import Config

import random


class Generation:
    numBlueprints = 1
    numModules = 5

    def __init__(self):
        self.speciesNumbers = []
        self.module_population, self.blueprint_population, self.da_population = None, None, None
        self.initialise_populations()

    def initialise_populations(self):
        # Picking the ranking function
        rank_fn = single_objective_rank if Config.second_objective == '' else cdn_rank

        self.module_population = Population(PopInit.initialise_modules(),
                                            rank_fn,
                                            PopInit.initialize_mutations(),
                                            Props.MODULE_POP_SIZE,
                                            2,
                                            2,
                                            Props.MODULE_TARGET_NUM_SPECIES)

        self.blueprint_population = Population(PopInit.initialise_blueprints(),
                                               rank_fn,
                                               PopInit.initialize_mutations(),
                                               Props.BP_POP_SIZE,
                                               2,
                                               2,
                                               Props.BP_TARGET_NUM_SPECIES)

        self.da_population = Population(PopInit.initialise_da(),
                                        rank_fn,
                                        PopInit.da_initial_mutations(),
                                        Props.DA_POP_SIZE,
                                        1,
                                        1,
                                        Props.DA_TARGET_NUM_SPECIES)

    def step(self):
        """Runs CDN for one generation - must be called after fitness evaluation"""
        self.module_population.step()
        for blueprint_individual in self.blueprint_population.individuals:
            blueprint_individual.reset_number_of_module_species(self.module_population.get_num_species())
        self.blueprint_population.step()
        self.da_population.step()

        for blueprint_individual in self.blueprint_population.individuals:
            blueprint_individual.end_step()

        for module_individual in self.module_population.individuals:
            module_individual.end_step()  # this also sets fitness to zero

    def evaluate(self, generation_number):
        print("Num species:", len(self.module_population.species))
        inputs, targets = Evaluator.sample_data('mnist')

        best_acc, best_second, best_third = float('-inf'), float('-inf'), float('-inf')
        best_bp, best_bp_genome = None, None
        accuracies, second_objective_values, third_objective_values = [], [], []

        # Randomize the list so that random individuals are sampled more often
        random.shuffle(self.blueprint_population.individuals)

        for i, blueprint_individual in enumerate(self.blueprint_population.individuals):
            # Checking if done enough evals
            if Props.INDIVIDUALS_TO_EVAL < i:
                break

            # All computationally expensive tests
            if Config.test_in_run:
                pass

            # Evaluating individual
            if Config.protect_parsing_from_errors:
                try:
                    module_graph, blueprint_individual, results = self.evaluate_blueprint(blueprint_individual, inputs,
                                                                                          generation_number)
                except Exception as e:
                    blueprint_individual.defective = True
                    print('Blueprint ran with errors, marking as defective\n', blueprint_individual)
                    print(e)
                    continue
            else:
                module_graph, blueprint_individual, results = self.evaluate_blueprint(blueprint_individual, inputs,
                                                                                      generation_number)
            # Gathering results
            acc = results[0]
            if len(results) >= 2:
                second = results[1]

                best_second = max(best_second, second)
                second_objective_values.append(second)

            if len(results) >= 3:
                third = results[3]

                best_third = max(best_third, third)
                third_objective_values.append(third)

            if len(results) > 3:
                raise Exception("Error: too many result values to unpack")

            # print("checking new net against previous")
            if acc >= best_acc:
                # print("found new best bp")
                best_acc = acc
                best_bp = module_graph
                best_bp_genome = blueprint_individual

            accuracies.append(acc)

        if generation_number % Config.print_best_graph_every_n_generations == 0:
            if Config.save_best_graphs:
                # print('Best blueprint:\n', best_bp_genome)
                best_bp.plot_tree_with_graphvis(title="gen:" + str(generation_number) + " acc:" + str(best_acc),
                                                file="best_of_gen_" + repr(generation_number))

        RuntimeAnalysis.log_new_generation(accuracies, generation_number,
                                           second_objective_values=(
                                               second_objective_values if len(second_objective_values) > 0 else None),
                                           third_objective_values=(
                                               third_objective_values if len(third_objective_values) > 0 else None))
        print('Best accuracy:', best_acc)

    def evaluate_blueprint(self, blueprint_individual, inputs, generation_number):
        blueprint = blueprint_individual.to_blueprint()
        # module_graph = blueprint.parseto_module_graph(self)
        module_graph, sans_aggregators = blueprint.parseto_module_graph(self, return_graph_without_aggregators=True)

        if module_graph is None:
            raise Exception("None module graph produced from blueprint")

        try:
            # print("using infeatures = ",module_graph.get_first_feature_count(inputs))
            net = module_graph.to_nn(in_features=module_graph.get_first_feature_count(inputs))
        except Exception as e:
            if Config.save_failed_graphs:
                module_graph.plot_tree_with_graphvis("module graph which failed to parse to nn")
            raise Exception("Error: failed to parse module graph into nn", e)

        net.specify_dimensionality(inputs)
        # try:
        #     net.specify_dimensionality(inputs)
        # except Exception as e:
        #     print("Error:", e)
        #     if Config.print_failed_graphs:
        #         module_graph.plot_tree_with_graphvis(title="module graph with error passing input through net",
        #                                              file="module_graph_with_agg")
        #         sans_aggregators.plot_tree_with_graphvis(title="previous module graph but without agg nodes",
        #                                                  file="module_graph_without_agg")
        #         print('failed graph:', blueprint_individual)
        #     raise Exception("Error: nn failed to have input passed through")

        if Config.dummy_run and generation_number < 50:
            acc = random.random()
            da_indv = blueprint_individual.pick_da_scheme(self.da_population)
            da_scheme = da_indv.to_phenotype()
        else:
            # TODO if DA on
            da_indv = blueprint_individual.pick_da_scheme(self.da_population)
            da_scheme = da_indv.to_phenotype()
            print("got da scheme from blueprint",da_scheme, "indv:",da_scheme)
            acc = Evaluator.evaluate(net, Config.number_of_epochs_per_evaluation, dataset='mnist', path='../../data',
                                     batch_size=256, augmentor=da_scheme)

        second_objective_value = None
        third_objective_value = None

        if Config.second_objective == "network_size":
            second_objective_value = 1 / net.module_graph.get_net_size()
        elif Config.second_objective == "":
            pass
        else:
            print("Error: did not recognise second objective", Config.second_objective)

        if second_objective_value is None:
            results = [acc]
        elif third_objective_value is None:
            results = acc, second_objective_value
        else:
            results = acc, second_objective_value, third_objective_value

        blueprint_individual.report_fitness(*results)
        for module_individual in blueprint_individual.modules_used:
            module_individual.report_fitness(*results)

        blueprint_individual.da_scheme.report_fitness(*results)

        return module_graph, blueprint_individual, results
