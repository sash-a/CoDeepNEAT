import src.Config.NeatProperties as Props
from src.NEAT.Population import Population, single_objective_rank, cdn_rank
from src.NeuralNetwork import Evaluator
from src.CoDeepNEAT import PopulationInitialiser as PopInit
from src.Analysis import RuntimeAnalysis
from src.Config import Config

import random
import multiprocessing as mp
import torch
import math
import sys


class Generation:
    numBlueprints = 1
    numModules = 5

    def __init__(self):
        self.speciesNumbers = []
        self.module_population, self.blueprint_population, self.da_population = None, None, None
        self.initialise_populations()

        self._bp_index = mp.Value('i', 0)

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

    def evaluate(self):
        procs = []
        results_dict = mp.Manager().dict()
        lock = mp.Lock()
        for i in range(Config.num_gpus):
            procs.append(mp.Process(target=self._evaluate, args=(lock, results_dict), name=str(i)))
            procs[-1].start()
            print('Started proc:', procs[-1])

        for proc in procs:
            proc.join()

        print('values: ', results_dict.values())

        self._bp_index.value = 0

    def _evaluate(self, lock, result_dict):
        inputs, targets = Evaluator.sample_data(Config.get_device())

        # TODO make global
        blueprints = self.blueprint_population.individuals
        bp_pop_size = len(blueprints)

        while self._bp_index.value < Props.INDIVIDUALS_TO_EVAL:
            with lock:
                blueprint_individual = blueprints[self._bp_index.value % bp_pop_size]
                self._bp_index.value += 1
                print('Proc:', mp.current_process().name, 'is evaluating bp', self._bp_index.value)

            print('lock released on', mp.current_process())
            # Evaluating individual
            try:
                module_graph, blueprint_individual, results = self.evaluate_blueprint(blueprint_individual, inputs)
                print('Eval done acc:', results[0], 'on proc:', mp.current_process().name, '\n\n')
                if mp.current_process().name in result_dict:
                    old_bp, old_results = result_dict[mp.current_process().name]
                    if results[0] > old_results[0]:  # TODO this only checks acc
                        result_dict[mp.current_process().name] = (blueprint_individual, results)
                else:
                    result_dict[mp.current_process().name] = (blueprint_individual, results)

            except Exception as e:
                blueprint_individual.defective = True

                if Config.protect_parsing_from_errors:
                    print('Blueprint ran with errors, marking as defective\n', blueprint_individual)
                    print(e)
                else:
                    raise Exception(e)

    def evaluate_blueprint(self, blueprint_individual, inputs):
        print('in eval on:', mp.current_process())
        blueprint = blueprint_individual.to_blueprint()
        module_graph, sans_aggregators = blueprint.parseto_module_graph(self, return_graph_without_aggregators=True)

        if module_graph is None:
            raise Exception("None module graph produced from blueprint")

        try:
            net = module_graph.to_nn(in_features=module_graph.get_first_feature_count(inputs)).to(Config.get_device())
            print('is now a net', mp.current_process())
            net.share_memory()
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
            acc = Evaluator.evaluate(net, Config.number_of_epochs_per_evaluation, Config.get_device(), batch_size=256,
                                     augmentor=da_scheme)
        print('run finished', mp.current_process())

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

        print('stuff is reported', mp.current_process())
        return module_graph, blueprint_individual, results


if __name__ == '__main__':
    pass
