import copy
import sys
import math
import random
import time

import multiprocessing as mp
import cv2

from Phenotype2.NeuralNetwork import Network

from data import DataManager

import src.Config.NeatProperties as Props
from src.Config import Config
from src.Validation import DataLoader, Validation
from src.Analysis import Logger

from src.CoDeepNEAT import PopulationInitialiser as PopInit
from src.NEAT.Population import Population
from src.NEAT.PopulationRanking import single_objective_rank, cdn_rank, nsga_rank
from src.Phenotype.ParetoPopulation import ParetoPopulation

from src.CoDeepNEAT.CDNGenomes.ModuleGenome import ModuleGenome
from src.CoDeepNEAT.CDNGenomes.BlueprintGenome import BlueprintGenome
from src.CoDeepNEAT.CDNGenomes.DAGenome import DAGenome

from src.CoDeepNEAT.CDNNodes.ModuleNode import ModuleNEATNode
from src.CoDeepNEAT.CDNNodes.BlueprintNode import BlueprintNEATNode
from src.CoDeepNEAT.CDNNodes.DANode import DANode
from src.NEAT.Species import Species

import wandb

"""the generation class is a container for the 3 cdn populations.
    It is also responsible for stepping the evolutionary cycle.
    The evaluation of blueprints, and its parallelisation is controlled by this class.
"""


class Generation:
    def __init__(self):
        self.speciesNumbers = []
        self.module_population, self.blueprint_population, self.da_population = None, None, None
        self.initialise_populations()
        self.generation_number = -1
        self.pareto_population = ParetoPopulation()

        tags = []
        if Config.module_retention:
            tags.append('module retention')
        if Config.speciation_overhaul:
            tags.append('speciation overhaul')
        if Config.evolve_data_augmentations:
            tags.append('evolve DA')

        if not tags:
            tags = ['base']

        # wandb.init(name=Config.run_name, project='cdn_test', tags=tags, dir='../../results')
        # wandb.config.module_retention = Config.module_retention
        # wandb.config.dataset = Config.dataset
        # wandb.config.evolution_epochs = Config.number_of_epochs_per_evaluation
        # wandb.config.new_speciation = Config.speciation_overhaul
        # wandb.config.da = Config.evolve_data_augmentations

    def initialise_populations(self):
        """starts off the populations of a new generation"""

        if Config.deterministic_pop_init:
            """to make the population initialisation deterministic"""
            random.seed(1)

        self.module_population = Population(
            PopInit.initialize_pop(ModuleNEATNode, ModuleGenome, Props.MODULE_POP_SIZE, True),
            None,
            PopInit.initialize_mutations(True),
            Props.MODULE_POP_SIZE,
            2,
            2,
            Props.MODULE_TARGET_NUM_SPECIES)

        self.blueprint_population = Population(
            PopInit.initialize_pop(BlueprintNEATNode, BlueprintGenome, Props.BP_POP_SIZE, True,
                                   self.module_population.individuals),
            None,
            PopInit.initialize_mutations(True),
            Props.BP_POP_SIZE,
            2,
            2,
            Props.BP_TARGET_NUM_SPECIES)

        if Config.evolve_data_augmentations:
            self.da_population = Population(
                PopInit.initialize_pop(DANode, DAGenome, Props.DA_POP_SIZE, False),
                single_objective_rank,
                PopInit.initialize_mutations(False),
                Props.DA_POP_SIZE,
                1,
                0,
                Props.DA_TARGET_NUM_SPECIES)

        if Config.deterministic_pop_init:
            """to make the rest of the evolutionary run random again"""
            random.seed()
        self.update_rank_function()

    def update_rank_function(self):
        """choses the populations ranking functions based on Config options.
            assigns the ranking functions to the populations"""

        rank_fn = single_objective_rank if Config.second_objective == '' else (
            cdn_rank if Config.moo_optimiser == "cdn" else nsga_rank)

        self.blueprint_population.rank_population_fn = rank_fn
        self.module_population.rank_population_fn = rank_fn

        num_objectives = (1 if Config.second_objective == '' else 2)

        all_indvs = self.blueprint_population.individuals
        all_indvs.extend(self.module_population.individuals)

        bad_init = sys.maxsize

        for indv in all_indvs:
            """update the fitness value arrays of individuals. 
            this should only be necessary if the moo options are changed mid run time
            for example when the algorithm is changed from single objective to multi objective mid run"""
            if len(indv.fitness_values) > num_objectives:
                """must drop a value, ie go from storing acc,complexity to just acc"""
                indv.fitness_values = indv.fitness_values[:num_objectives]

            while len(indv.fitness_values) < num_objectives:
                """must prepare for a new fitness score: complexity"""
                indv.fitness_values.append(bad_init)

    def step(self):
        """Runs CDN for one generation - must be called after fitness evaluation"""
        self.pareto_population.update_pareto_front()

        self.module_population.step_evolution(self)
        for blueprint_individual in self.blueprint_population.individuals:
            blueprint_individual.reset_number_of_module_species(self.module_population.get_num_species(),
                                                                self.generation_number)

        self.blueprint_population.step_evolution(self)

        if Config.evolve_data_augmentations:
            self.da_population.step_evolution(self)

        for blueprint_individual in self.blueprint_population.individuals:
            blueprint_individual.end_step(self)

        for module_individual in self.module_population.individuals:
            module_individual.end_step()  # this also sets fitness to zero

        DataManager.save_generation_state(self)

        print('Module species distribution:', ', '.join([str(len(spc)) for spc in self.module_population.species]))

    def evaluate(self, generation_number):
        """evaluates all blueprints multiple times.
            passes evaluation scores back to individuals
        """

        self.generation_number = generation_number

        procs = []
        manager = mp.Manager()
        results_dict = manager.dict()
        bp_index = manager.Value('i', 0)
        lock = mp.Lock()

        if Config.evolve_data_augmentations:
            # blueprints should pick their da schemes before being evaluated. if their old DA is still alive they
            # will reselect it
            for blueprint_individual in self.blueprint_population.individuals:
                blueprint_individual.pick_da_scheme(self.da_population)

        for i in range(Config.num_gpus):
            procs.append(mp.Process(target=self._evaluate, args=(lock, bp_index, results_dict), name=str(i)))
            procs[-1].start()
        [proc.join() for proc in procs]

        accuracies, second_objective_values, third_objective_values = [], [], []
        bp_pop_size = len(self.blueprint_population)
        bp_pop_indvs = self.blueprint_population.individuals

        # new_accs, new_accs_graph = [], []
        # acc_table = wandb.Table(columns=["old accuracy", "new accuracy", "new accuracy graph"])
        # time_table = wandb.Table(columns=["old time", "new time", "new time graph"])
        # TODO log image of best graph: wandb.log({"examples": [wandb.Image(numpy_array_or_pil, caption="Label")]})

        for bp_key, (fitness, evaluated_bp, module_graph, new_acc, new_acc_graph, new_train_time, new_train_time_graph,
                     old_train_time) in results_dict.items():
            if fitness == 'defective':
                bp_pop_indvs[bp_key % bp_pop_size].defective = True
                continue

            # pheno comparison stuff
            # acc_table.add_data(fitness[0], new_acc, new_acc_graph)
            # time_table.add_data(old_train_time, new_train_time, new_train_time_graph)
            #
            # new_accs.append(new_acc)
            # new_accs_graph.append(new_accs_graph)

            # Validation
            if evaluated_bp.eq(bp_pop_indvs[bp_key % bp_pop_size]):
                raise Exception('Evaluated bp topology not same as main one')
            if not evaluated_bp.modules_used_index:
                raise Exception('Modules used index is empty in evaluated bp', evaluated_bp.modules_used_index)
            if not evaluated_bp.modules_used:
                raise Exception('Modules used is empty in evaluated bp', evaluated_bp.modules_used)

            # Fitness assignment
            bp_pop_indvs[bp_key % bp_pop_size].report_fitness(fitness)

            da_scheme_inherited = None if not (Config.allow_da_scheme_ignores and Config.evolve_data_augmentations) \
                else self.da_population[evaluated_bp.da_scheme_index]
            bp_pop_indvs[bp_key % bp_pop_size]. \
                inherit_species_module_mapping(self,
                                               evaluated_bp,
                                               fitness[0],
                                               da_scheme=da_scheme_inherited,
                                               inherit_module_mapping=Config.module_retention)

            if Config.evolve_data_augmentations and evaluated_bp.da_scheme_index != -1:
                self.da_population[evaluated_bp.da_scheme_index].report_fitness([fitness[0]])

            for species_index, member_index in evaluated_bp.modules_used_index:
                if Config.second_objective == "":
                    if isinstance(member_index, tuple):
                        spc, mod = member_index
                        self.module_population.species[spc][mod].report_fitness(fitness)
                    else:
                        self.module_population.species[species_index][member_index].report_fitness(fitness)
                else:
                    if isinstance(member_index, tuple):
                        spc, mod = member_index
                        module_indv = self.module_population.species[spc][mod]
                    else:
                        module_indv = self.module_population.species[species_index][member_index]

                    acc = fitness[0]

                    if Config.second_objective == 'network_size':
                        comp = module_indv.get_comlexity()
                    elif Config.second_objective == 'network_size_adjusted':
                        comp = module_indv.get_comlexity() / pow(acc, 2)
                    elif Config.second_objective == 'network_size_adjusted_2':
                        comp = pow(module_indv.get_comlexity(), 0.5) / pow(acc, 2)
                    else:
                        raise Exception()
                    module_indv.report_fitness([acc, comp])

            # Gathering results for analysis
            accuracies.append(fitness[0])
            if len(fitness) > 1:
                second_objective_values.append(fitness[1])
            if len(fitness) > 2:
                third_objective_values.append(fitness[2])

            self.pareto_population.queue_candidate(module_graph)

        # avg_old_acc = sum(accuracies) / len(accuracies)
        # avg_new_acc = sum(new_accs) / len(new_accs)
        # avg_nag = sum(new_accs_graph) / len(new_accs_graph)
        #
        # wandb.log({'accuracies': acc_table, 'time': time_table, 'old accuracies': accuracies,
        #            'average old accuracy': avg_old_acc, 'average new accuracies': avg_new_acc,
        #            'average new accuracies with graph': avg_nag},
        #           step=generation_number)

        Logger.log_new_generation(accuracies, generation_number,
                                  second_objective_values=(
                                      second_objective_values if second_objective_values else None),
                                  third_objective_values=(
                                      third_objective_values if third_objective_values else None))

    def _evaluate(self, lock, bp_index, result_dict):
        cv2.setNumThreads(0)

        inputs, targets = DataLoader.sample_data(Config.get_device())
        blueprints = self.blueprint_population.individuals
        bp_pop_size = len(blueprints)

        while bp_index.value < Props.INDIVIDUALS_TO_EVAL:
            with lock:
                blueprint_individual = copy.deepcopy(blueprints[bp_index.value % bp_pop_size])
                curr_index = bp_index.value
                bp_index.value += 1

            # Evaluating individual
            module_graph, blueprint_individual, results, new_acc, new_acc_graph, new_train_time, new_train_time_graph, old_train_time = self.evaluate_blueprint(
                blueprint_individual, inputs, curr_index)
            result_dict[
                curr_index] = results, blueprint_individual, module_graph, new_acc, new_acc_graph, new_train_time, new_train_time_graph, old_train_time

    def evaluate_blueprint(self, blueprint_individual, inputs, index):
        blueprint_individual: BlueprintGenome
        # Validation
        if blueprint_individual.modules_used_index:
            raise Exception('Modules used index is not empty', blueprint_individual.modules_used_index)
        if blueprint_individual.modules_used:
            raise Exception('Modules used is not empty', blueprint_individual.modules_used)

        bpcp = copy.deepcopy(blueprint_individual)

        # Testing old
        s_constr = time.time()
        blueprint_graph = blueprint_individual.to_blueprint()
        module_graph = blueprint_graph.parse_to_module_graph(self,
                                                             allow_ignores=True if index >= Props.BP_POP_SIZE else False)
        blueprint_individual: BlueprintGenome

        if index == 1:
            blueprint_individual.plot_tree_with_graphvis(view=True, file='bp')
            for i, (spc, mod) in enumerate(blueprint_individual.species_module_index_map.items()):
                module = self.module_population.species[spc][mod]
                module.plot_tree_with_graphvis(view=True, file='mod' + str(i))

            module_graph.plot_tree_with_graphvis(view=True)

        net = Validation.create_nn(module_graph, inputs)
        old_construction_time = time.time() - s_constr

        bpcp.species_module_index_map = blueprint_individual.species_module_index_map

        Config.use_graph = True
        bp = copy.deepcopy(bpcp)
        s_constr = time.time()
        new_net = Network(bp, self.module_population.species, list(inputs.size())).to(Config.get_device())
        new_construction_time = time.time() - s_constr

        # Visualizing stuff
        # blueprint_individual.plot_tree_with_graphvis(view=True, file='bp')
        # blueprint_individual.modules_used[0].plot_tree_with_graphvis(view=True, file='mod')
        # net.module_graph.plot_tree_with_graphvis(title='old', view=True, file='old')
        # new_net.visualize(view=True)

        # Number of parameters
        # par = list(net.module_graph.module_graph_root_node.get_parameters({}))
        # par.extend(net.final_layer.parameters())
        # import numpy as np
        # model_parameters = filter(lambda p: p.requires_grad, par)
        # params = sum([np.prod(p.size()) for p in model_parameters])
        # print('old', params)
        # model_parameters = filter(lambda p: p.requires_grad, new_net.parameters())
        # params = sum([np.prod(p.size()) for p in model_parameters])
        # print('new', params)

        if Config.evolve_data_augmentations:
            if Config.allow_da_scheme_ignores and random.random() < Config.da_ignore_chance:
                # ignore da scheme to try different one
                blueprint_individual.da_scheme = None
                da_indv = blueprint_individual.pick_da_scheme(self.da_population)
            else:
                # use existing da scheme
                da_indv = blueprint_individual.da_scheme

            if da_indv.has_branches():
                da_indv.plot_tree_with_graphvis()
                raise Exception('Found data augmentation with branches, this should not happen')

            da_scheme = da_indv.to_phenotype()
            module_graph.data_augmentation_schemes.append(da_indv)
        else:
            da_scheme = None

        # Testing old train time
        s_train = time.time()
        accuracy = Validation.get_accuracy_estimate_for_network(net, da_scheme=da_scheme, batch_size=Config.batch_size)
        old_train_time = time.time() - s_train

        # -------------------Testing new phenotype------------------------------

        # Creating the network with the same modules as used previously
        Config.use_graph = False
        n2 = Network(copy.deepcopy(bpcp), self.module_population.species, list(inputs.size())).to(Config.get_device())
        s_train = time.time()
        new_acc = Validation.get_accuracy_estimate_for_network(n2, da_scheme=None, batch_size=Config.batch_size)
        new_train_time = time.time() - s_train

        Config.use_graph = True
        n3 = Network(bpcp, self.module_population.species, list(inputs.size())).to(Config.get_device())
        s_train = time.time()
        new_acc_graph = Validation.get_accuracy_estimate_for_network(n3, da_scheme=None, batch_size=Config.batch_size)
        new_train_time_graph = time.time() - s_train

        with open('traintime.txt', 'a+') as f:
            f.write('\nnew:' + str(new_train_time))
            f.write('\nnew_graph:' + str(new_train_time_graph))
            f.write('\nold:' + str(old_train_time))

        with open('constructiontime.txt', 'a+') as f:
            f.write('\nnew:' + str(new_construction_time))
            f.write('\nold:' + str(old_construction_time))

        with open('acc.txt', 'a+') as f:
            f.write('\nnew:' + str(new_acc))
            f.write('\nnew_graph:' + str(new_acc_graph))
            f.write('\nold:' + str(accuracy))

        # End of logging

        objective_names = [Config.second_objective, Config.third_objective]
        results = [accuracy]
        for objective_name in objective_names:
            if objective_name == 'network_size':
                results.append(net.module_graph.get_net_size())
            elif objective_name == 'network_size_adjusted':
                results.append(net.module_graph.get_net_size() / (accuracy * accuracy))
            elif objective_name == 'network_size_adjusted_2':
                results.append(math.sqrt(net.module_graph.get_net_size()) / (accuracy * accuracy))
            elif objective_name == '':
                pass
            else:
                print("Error: did not recognise second objective", Config.second_objective)

        module_graph.delete_all_layers()
        module_graph.fitness_values = results

        return module_graph, blueprint_individual, results, new_acc, new_acc_graph, new_train_time, new_train_time_graph, old_train_time

    def get_topology_mutation_modifier(self):
        """for the global mutation magnitide extension.
            returns: the top mod as a function of the generation number
        """
        return self._get_mutation_modifier(3, 7.5, 3.5)

    def get_attribute_mutation_modifier(self):
        """for the global mutation magnitide extension.
          returns: the att mod as a function of the generation number
                """
        return self._get_mutation_modifier(6.2, 10, 4)

    def _get_mutation_modifier(self, a, b, c):
        completion_frac = self.generation_number / Config.max_num_generations
        return math.atan(a - b * completion_frac) / c + 0.9
