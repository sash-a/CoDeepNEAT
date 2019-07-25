import math

import src.Config.NeatProperties as Props
from src.NEAT.Population import Population
from src.NEAT.PopulationRanking import single_objective_rank, cdn_rank, nsga_rank
from src.NeuralNetwork import Evaluator
from src.CoDeepNEAT import PopulationInitialiser as PopInit
from src.Analysis import RuntimeAnalysis
from src.Config import Config
from data import DataManager
from src.NeuralNetwork.ParetoPopulation import ParetoPopulation
import pickle

import multiprocessing as mp
import random
import torch


class Generation:

    def __init__(self):
        self.speciesNumbers = []
        self.module_population, self.blueprint_population, self.da_population = None, None, None
        self.initialise_populations()
        self.generation_number = -1
        self.pareto_population = ParetoPopulation()

    def initialise_populations(self):
        # Picking the ranking function
        rank_fn = single_objective_rank if Config.second_objective == '' else (
            cdn_rank if Config.moo_optimiser == "cdn" else nsga_rank())

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

        if Config.evolve_data_augmentations:
            self.da_population = Population(PopInit.initialise_da(),
                                            rank_fn,
                                            PopInit.da_initial_mutations(),
                                            Props.DA_POP_SIZE,
                                            1,
                                            1,
                                            Props.DA_TARGET_NUM_SPECIES)

    def step(self):
        """Runs CDN for one generation - must be called after fitness evaluation"""
        self.pareto_population.update_pareto_front()
        # self.pareto_population.plot_fitnesses()
        self.module_population.step()
        for blueprint_individual in self.blueprint_population.individuals:
            print('bp fitness:', blueprint_individual.fitness_values)
            blueprint_individual.reset_number_of_module_species(self.module_population.get_num_species())

        self.blueprint_population.step()

        if Config.evolve_data_augmentations:
            self.da_population.step()

        for blueprint_individual in self.blueprint_population.individuals:
            blueprint_individual.end_step()

        for module_individual in self.module_population.individuals:
            module_individual.end_step()  # this also sets fitness to zero

        DataManager.save_generation_state(self)

    def evaluate(self, generation_number):
        self.generation_number = generation_number

        import copy
        reps = math.ceil(Props.INDIVIDUALS_TO_EVAL / len(self.blueprint_population.individuals))
        blueprints = [copy.deepcopy(bp) for _ in range(reps) for bp in self.blueprint_population.individuals]

        # TODO test if works with > 1 species
        # blueprints = self.blueprint_population.individuals * math.ceil(
        #     Props.INDIVIDUALS_TO_EVAL / len(self.blueprint_population.individuals))
        # random.shuffle(blueprints)
        blueprints = blueprints[:Props.INDIVIDUALS_TO_EVAL]

        if not Config.is_parallel():
            evaluations = []
            for bp in blueprints:
                evaluations.append(self.evaluate_blueprint(bp))
        else:
            pool = mp.Pool(Config.num_gpus)
            evaluations = pool.imap(self.evaluate_blueprint, blueprints)

        accuracies, second_objective_values, third_objective_values = [], [], []

        bp_pop_indv = self.blueprint_population.individuals
        bp_pop_len = len(bp_pop_indv)
        for bp_key, evaluation in enumerate(evaluations):
            if evaluation is None:
                bp_pop_indv[bp_key % bp_pop_len].defective = True
                continue

            # TODO test this extensively (from Shane)
            evaluated_bp, fitness, module_graph = evaluation
            bp_pop_indv[bp_key % bp_pop_len].report_fitness(*fitness)

            # print('eval', evaluated_bp)
            # print('real', bp_pop_indv[bp_key % bp_pop_len])

            if evaluated_bp.eq(bp_pop_indv[bp_key % bp_pop_len]):
                raise Exception('Evaled bp topology not same as main one')

            if Config.evolve_data_augmentations and evaluated_bp.da_scheme_index != -1:
                self.da_population[evaluated_bp.da_scheme_index].report_fitness(*fitness)

            if not evaluated_bp.modules_used_index:
                raise Exception('Modules used index is empty in evaluated bp', evaluated_bp.modules_used_index)

            if not evaluated_bp.modules_used:
                raise Exception('Modules used is empty in evaluated bp', evaluated_bp.modules_used)

            for species_index, member_index in evaluated_bp.modules_used_index:
                self.module_population.species[species_index][member_index].report_fitness(*fitness)

            accuracies.append(fitness[0])
            if len(fitness) > 1:
                second_objective_values.append(fitness[1])
            if len(fitness) > 2:
                third_objective_values.append(fitness[2])

            self.pareto_population.queue_candidate(module_graph)

        RuntimeAnalysis.log_new_generation(accuracies, generation_number,
                                           second_objective_values=(
                                               second_objective_values if second_objective_values else None),
                                           third_objective_values=(
                                               third_objective_values if third_objective_values else None))

    def evaluate_blueprint(self, blueprint_individual):
        try:
            device = Config.get_device()
            inputs, _ = Evaluator.sample_data(device)

            if blueprint_individual.modules_used_index:
                raise Exception('Modules used index is not empty', blueprint_individual.modules_used_index)

            if blueprint_individual.modules_used:
                raise Exception('Modules used is not empty', blueprint_individual.modules_used)

            blueprint = blueprint_individual.to_blueprint()
            module_graph, sans_aggregators = blueprint.parseto_module_graph(self, return_graph_without_aggregators=True)
            if module_graph is None:
                raise Exception("None module graph produced from blueprint")
            try:
                net = module_graph.to_nn(in_features=module_graph.get_first_feature_count(inputs)).to(device)
            except Exception as e:
                if Config.save_failed_graphs:
                    module_graph.plot_tree_with_graphvis("module graph which failed to parse to nn")
                raise Exception("Error: failed to parse module graph into nn", e)

            net.configure(blueprint_individual.learning_rate(), blueprint_individual.beta1(),
                          blueprint_individual.beta2())
            net.specify_dimensionality(inputs)

            if Config.dummy_run:
                acc = hash(net)
                if Config.evolve_data_augmentations:
                    da_indv = blueprint_individual.pick_da_scheme(self.da_population)
                    da_scheme = da_indv.to_phenotype()
                    # da_indv.plot_tree_with_graphvis("test")
                    # print('THISSSSSSSSSSSSSSSSSSS', da_indv)
                else:
                    da_scheme = None
            else:
                if Config.evolve_data_augmentations:
                    da_indv = blueprint_individual.pick_da_scheme(self.da_population)
                    da_scheme = da_indv.to_phenotype()
                    # da_indv.plot_tree_with_graphvis("test")
                else:
                    da_scheme = None
                # print("got da scheme from blueprint", da_scheme, "indv:", da_scheme)

                acc = Evaluator.evaluate(net, Config.number_of_epochs_per_evaluation, device, 256, augmentor=da_scheme)

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

            module_graph.delete_all_layers()

            return blueprint_individual, results, module_graph
        except Exception as e:
            if not Config.protect_parsing_from_errors:
                raise Exception(e)

            print('Blueprint ran with errors, marking as defective\n', blueprint_individual)
            print(e)
            return None
