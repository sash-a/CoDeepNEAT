import json
import operator
from configparser import ConfigParser

import torch
import torch.multiprocessing as mp


def load(file):
    print('Loading config from:', file)

    parser = ConfigParser()
    parser.read(file)

    print('Config options in', file)
    for section in parser.sections():
        print(section)
        print(json.dumps(dict(parser[section]), indent=4))

    if 'run' in parser:
        run = parser['run']
        Config.run_name = run.get('name', Config.run_name)
        Config.continue_from_last_run = run.getboolean('continue_run', Config.continue_from_last_run)
        Config.max_num_generations = run.getint('generations', Config.max_num_generations)

        # print('run:', Config.run_name, Config.continue_from_last_run, Config.max_num_generations)

    if 'modules' in parser:
        modules = parser['modules']
        Config.module_pop_size = modules.getint('population_size', Config.module_pop_size)
        Config.module_node_mutation_chance = modules.getfloat('prob_add_node', Config.module_node_mutation_chance)
        Config.module_conn_mutation_chance = modules.getfloat('prob_add_connection', Config.module_conn_mutation_chance)
        Config.module_target_num_species = modules.getint('target_num_species', Config.module_target_num_species)

    if 'blueprints' in parser:
        bps = parser['blueprints']
        Config.bp_pop_size = bps.getint('population_size', Config.bp_pop_size)
        Config.bp_node_mutation_chance = bps.getfloat('prob_add_node', Config.bp_node_mutation_chance)
        Config.bp_conn_mutation_chance = bps.getfloat('prob_add_connection', Config.bp_conn_mutation_chance)
        Config.bp_target_num_species = bps.getint('target_num_species', Config.bp_target_num_species)

    if 'data_augmentations' in parser:
        das = parser['data_augmentations']
        Config.evolve_data_augmentations = das.getboolean('evolve', Config.evolve_data_augmentations)
        Config.colour_augmentations = das.getboolean('colour', Config.colour_augmentations)
        Config.da_pop_size = das.getint('population_size', Config.da_pop_size)
        Config.da_node_mutation_chance = das.getfloat('mutation_chance', Config.da_node_mutation_chance)
        Config.da_target_num_species = das.getint('target_num_species', Config.da_target_num_species)

        # print('DA:', Config.evolve_data_augmentations, Config.colour_augmentations, Config.da_pop_size,
        #       Config.da_node_mutation_chance, Config.da_target_num_species)

    if 'species' in parser:
        spc = parser['species']
        Config.species_distance_thresh = spc.getfloat('distance_thresh', Config.species_distance_thresh)
        Config.species_distance_thresh_mod_min = spc.getfloat('distance_thresh_mod_min',
                                                              Config.species_distance_thresh_mod_min)
        Config.species_distance_thresh_mod_base = spc.getfloat('distance_thresh_mod_base',
                                                               Config.species_distance_thresh_mod_base)
        Config.species_distance_thresh_mod_max = spc.getfloat('distance_thresh_mod_max',
                                                              Config.species_distance_thresh_mod_max)
        Config.percent_to_reproduce = spc.getfloat('percent_to_reproduce', Config.percent_to_reproduce)
        Config.elite_to_keep = spc.getfloat('elite_to_keep', Config.elite_to_keep)

        # print('Species:', Config.species_distance_thresh, Config.species_distance_thresh_mod_min,
        #       Config.species_distance_thresh_mod_base, Config.species_distance_thresh_mod_max,
        #       Config.percent_to_reproduce, Config.elite_to_keep)

    if 'crossover' in parser:
        crsvr = parser['crossover']
        Config.excess_coefficient = crsvr.getfloat('excess_coefficient', Config.excess_coefficient)
        Config.disjoint_coefficient = crsvr.getfloat('disjoint_coefficient', Config.disjoint_coefficient)
        Config.layer_size_coefficient = crsvr.getfloat('layer_size_coefficient', Config.layer_size_coefficient)
        Config.layer_type_coefficient = crsvr.getfloat('layer_type_coefficient', Config.layer_type_coefficient)

        # print('Crossover:', Config.excess_coefficient, Config.disjoint_coefficient,
        #       Config.layer_size_coefficient, Config.layer_type_coefficient)

    if 'nn' in parser:
        nn = parser['nn']
        Config.device = nn.get('device', Config.device)
        Config.num_gpus = nn.getint('gpus', Config.num_gpus)
        Config.num_workers = nn.getint('num_workers', Config.num_workers)
        Config.dataset = nn.get('dataset', Config.dataset)
        Config.data_path = nn.get('data_path', Config.data_path)
        Config.number_of_epochs_per_evaluation = nn.getint('epochs', Config.number_of_epochs_per_evaluation)

        # print('nn:', Config.device, Config.num_gpus, Config.num_workers, Config.dataset, Config.data_path,
        #       Config.number_of_epochs_per_evaluation)

    if 'objectives' in parser:
        objs = parser['objectives']
        Config.second_objective = objs.get('second_objective', Config.second_objective)
        scnd_comp = objs.get('second_objective_comparator', 'lt')
        Config.second_objective_comparator = operator.lt if scnd_comp == 'lt' else operator.gt

        Config.third_objective = objs.get('third_objective', Config.third_objective)
        third_comp = objs.get('third_objective_comparator', 'lt')
        Config.third_objective_comparator = operator.lt if third_comp == 'lt' else operator.gt

        # print('Objectives:', Config.second_objective, Config.second_objective_comparator, Config.third_objective,
        #       Config.third_objective_comparator)

    if 'extensions' in parser:
        exts = parser['extensions']
        Config.moo_optimiser = exts.get('moo_optimiser', Config.moo_optimiser)
        Config.maintain_module_handles = exts.get('maintain_module_handles', Config.maintain_module_handles)
        Config.speciation_overhaul = exts.get('speciation_overhaul', Config.speciation_overhaul)
        Config.ignore_disabled_connections_for_topological_similarity = exts.get(
            'ignore_disabled_connections_for_topological_similarity',
            Config.ignore_disabled_connections_for_topological_similarity)

        # print('Extensions:', Config.moo_optimiser, Config.maintain_module_handles, Config.speciation_overhaul,
        #       Config.ignore_disabled_connections_for_topological_similarity)

    if 'test' in parser:
        test = parser['test']
        Config.dummy_run = test.getboolean('dummy_run', Config.dummy_run)
        Config.protect_parsing_from_errors = test.getboolean('protect_parsing_from_errors',
                                                             Config.protect_parsing_from_errors)
        Config.test_in_run = test.getboolean('test_in_run', Config.test_in_run)
        Config.interleaving_check = test.getboolean('interleaving_check', Config.interleaving_check)

        Config.save_best_graphs = test.getboolean('save_best_graphs', Config.save_best_graphs)
        Config.print_best_graphs = test.getboolean('print_best_graphs', Config.print_best_graphs)
        Config.print_best_graph_every_n_generations = test.getint('print_best_graph_every_n_generations',
                                                                  Config.print_best_graph_every_n_generations)
        Config.save_failed_graphs = test.getboolean('save_failed_graphs', Config.save_failed_graphs)

        # print('Test:', Config.dummy_run, Config.protect_parsing_from_errors, Config.test_in_run,
        #       Config.interleaving_check, Config.save_best_graphs, Config.print_best_graphs,
        #       Config.print_best_graph_every_n_generations, Config.save_failed_graphs)


class Config:
    # -----------------------------------------------------------------------------------------------------------------#
    # Run state options
    run_name = 'base_final'
    continue_from_last_run = True
    max_num_generations = 25
    # -----------------------------------------------------------------------------------------------------------------#
    # Modules
    module_pop_size = 56
    module_node_mutation_chance = 0.12  # 0.08
    module_conn_mutation_chance = 0.12  # 0.08
    module_target_num_species = 4
    # -----------------------------------------------------------------------------------------------------------------#
    # Blueprints
    bp_pop_size = 22
    bp_node_mutation_chance = 0.16
    bp_conn_mutation_chance = 0.12
    bp_target_num_species = 1
    # -----------------------------------------------------------------------------------------------------------------#
    # Data augmentations
    evolve_data_augmentations = True
    colour_augmentations = True
    da_pop_size = 20
    da_node_mutation_chance = 0.1
    da_target_num_species = 1
    # -----------------------------------------------------------------------------------------------------------------#
    individuals_to_eval = 88  # Number of times to sample blueprints
    mutation_tries = 100  # number of tries a mutation gets to pick acceptable individual
    # -----------------------------------------------------------------------------------------------------------------#
    # species
    species_distance_thresh = 1.5
    species_distance_thresh_mod_min = 0.001
    species_distance_thresh_mod_base = 0.1
    species_distance_thresh_mod_max = 100

    percent_to_reproduce = 0.2
    elite_to_keep = 0.1
    # -----------------------------------------------------------------------------------------------------------------#
    # Crossover
    excess_coefficient = 5
    disjoint_coefficient = 3
    layer_size_coefficient = 0  # 3
    layer_type_coefficient = 0  # 2
    # -----------------------------------------------------------------------------------------------------------------#
    # nn options
    device = 'gpu'  # gpu | cpu
    num_gpus = 4
    num_workers = 0  # this doesn't work in parallel because daemonic processes cannot spawn children
    dataset = 'cifar10'
    data_path = ''
    number_of_epochs_per_evaluation = 8
    # -----------------------------------------------------------------------------------------------------------------#
    # Objective options
    second_objective = 'network_size'  # network_size | network_size_adjusted | network_size_adjusted_2
    second_objective_comparator = operator.lt  # lt for minimisation, gt for maximisation
    third_objective = ''
    third_objective_comparator = operator.lt

    # -----------------------------------------------------------------------------------------------------------------#
    # Extensions
    moo_optimiser = 'cdn'  # cdn | nsga

    maintain_module_handles = False
    fitness_aggregation = "avg"  # max|avg

    speciation_overhaul = False
    ignore_disabled_connections_for_topological_similarity = False
    use_graph_edit_distance = False
    # -----------------------------------------------------------------------------------------------------------------#
    # Debug and test options
    dummy_run = False

    protect_parsing_from_errors = False
    test_in_run = False
    interleaving_check = False

    save_best_graphs = False
    print_best_graphs = False
    print_best_graph_every_n_generations = 5
    save_failed_graphs = False

    # -----------------------------------------------------------------------------------------------------------------#
    @classmethod
    def get_device(cls):
        """Used to obtain the correct device taking into account multiple GPUs"""
        gpu = 'cuda:'
        gpu += '0' if cls.num_gpus <= 1 else mp.current_process().name
        return torch.device('cpu') if cls.device == 'cpu' else torch.device(gpu)

    @classmethod
    def is_parallel(cls):
        return not (cls.device == 'cpu' or cls.num_gpus <= 1)
