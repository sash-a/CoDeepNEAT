import math

import src.Config.NeatProperties as Props
from src.NEAT.Population import Population, single_objective_rank, cdn_rank
from src.NeuralNetwork import Evaluator
from src.CoDeepNEAT import PopulationInitialiser as PopInit
from src.Analysis import RuntimeAnalysis
from src.Config import Config

import multiprocessing as mp
import random
import torch


class Generation:
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
            print('bp fitness:', blueprint_individual.fitness_values)
            blueprint_individual.reset_number_of_module_species(self.module_population.get_num_species())

        self.blueprint_population.step()
        self.da_population.step()

        for blueprint_individual in self.blueprint_population.individuals:
            blueprint_individual.end_step()

        for module_individual in self.module_population.individuals:
            module_individual.end_step()  # this also sets fitness to zero

    def evaluate(self):
        blueprints = self.blueprint_population.individuals * math.ceil(
            Props.INDIVIDUALS_TO_EVAL / len(self.blueprint_population.individuals))
        random.shuffle(blueprints)
        blueprints = blueprints[:Props.INDIVIDUALS_TO_EVAL]

        if not Config.is_parallel():
            evaluations = []
            for bp in blueprints:
                evaluations.append(self.evaluate_blueprint(bp))
        else:
            pool = mp.Pool(Config.num_gpus)
            evaluations = pool.imap(self.evaluate_blueprint, blueprints)

        bp_pop_size = len(self.blueprint_population)
        for bp_key, evaluation in enumerate(evaluations):
            if evaluation is None:
                self.blueprint_population[bp_key % bp_pop_size].defective = True
                continue

            evaluated_bp, fitness = evaluation
            self.blueprint_population[bp_key % bp_pop_size].report_fitness(*fitness)

            if evaluated_bp.da_scheme_index != -1:
                self.da_population[evaluated_bp.da_scheme_index].report_fitness(*fitness)
            for species_index, member_index in evaluated_bp.modules_used_index:
                self.module_population.species[species_index][member_index].report_fitness(*fitness)

    def evaluate_blueprint(self, blueprint_individual):
        try:
            device = Config.get_device()
            print('in eval', device)
            inputs, _ = Evaluator.sample_data(device)

            blueprint = blueprint_individual.to_blueprint()
            module_graph, sans_aggregators = blueprint.parseto_module_graph(self, return_graph_without_aggregators=True)
            if module_graph is None:
                raise Exception("None module graph produced from blueprint")
            try:
                # print("using infeatures = ",module_graph.get_first_feature_count(inputs))
                net = module_graph.to_nn(in_features=module_graph.get_first_feature_count(inputs)).to(device)
            except Exception as e:
                if Config.save_failed_graphs:
                    module_graph.plot_tree_with_graphvis("module graph which failed to parse to nn")
                raise Exception("Error: failed to parse module graph into nn", e)
            net.specify_dimensionality(inputs)

            if Config.dummy_run:
                acc = hash(net)
                da_indv = blueprint_individual.pick_da_scheme(self.da_population)
                da_scheme = da_indv.to_phenotype()
            else:
                # TODO if DA on
                da_indv = blueprint_individual.pick_da_scheme(self.da_population)
                da_scheme = da_indv.to_phenotype()
                acc = Evaluator.evaluate(net, Config.number_of_epochs_per_evaluation, device, 256, da_scheme)

            second_objective_value = None
            third_objective_value = None

            if Config.second_objective == "network_size":
                second_objective_value = net.module_graph.get_net_size()
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

            return blueprint_individual, results
        except Exception as e:
            if not Config.protect_parsing_from_errors:
                raise Exception(e)

            blueprint_individual.defective = True
            print('Blueprint ran with errors, marking as defective\n', blueprint_individual)
            print(e)
            return None
