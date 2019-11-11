import operator

import torch
import torch.multiprocessing as mp

use_graph = True
# --------------------------------------------------------------------------------------------------------------------#
# Run state options
run_name = "compare_test"
continue_from_last_run = False
deterministic_pop_init = True
dummy_run = False
max_num_generations = 30

# --------------------------------------------------------------------------------------------------------------------#
# nn options
device = 'gpu'  # gpu | cpu
num_gpus = 1
num_workers = 0  # this doesn't work in parallel because daemonic processes cannot spawn children
dataset = 'cifar10'
data_path = ''
number_of_epochs_per_evaluation = 5
batch_size = 256

# --------------------------------------------------------------------------------------------------------------------#
# fully train options
fully_train = False

num_epochs_in_full_train = 300
num_augs_in_full_train = 1
# multiplies feature count of every layer by this number to increase or decrease bandwidth
feature_multiplier_for_fully_train = 1

toss_bad_runs = False
drop_learning_rate = True
drop_period = 30
drop_factor = 1.2
use_adaptive_learning_rate_adjustment = False

# --------------------------------------------------------------------------------------------------------------------#
# Multiobjective options
second_objective = ''  # network_size | network_size_adjusted | network_size_adjusted_2
second_objective_comparator = operator.lt  # lt for minimisation, gt for maximisation
third_objective = ''
third_objective_comparator = operator.lt

moo_optimiser = 'cdn'  # cdn | nsga

# --------------------------------------------------------------------------------------------------------------------#
# Data augmentation options
evolve_data_augmentations = False

colour_augmentations = True
allow_da_scheme_ignores = True
da_ignore_chance = 0.2
train_on_origonal_data = True
batch_by_batch = False

# --------------------------------------------------------------------------------------------------------------------#
# module retention options
module_retention = True
fitness_aggregation = 'max'  # max | avg

allow_species_module_mapping_ignores = True
allow_cross_species_mappings = False
# --------------------------------------------------------------------------------------------------------------------#
# specitation options
speciation_overhaul = True

blueprint_nodes_use_representatives = False  # section 3.2.4 Sasha's paper
rep_mutation_chance_early = 0.6
rep_mutation_chance_late = 0.2
similar_rep_mutation_chance = 0.2  # chance to mutate all of the nodes with the same representative in the same way
closest_reps_to_consider = 6

use_graph_edit_distance = False
ignore_disabled_connections_for_topological_similarity = False

allow_attribute_distance = False
# --------------------------------------------------------------------------------------------------------------------#
# mutation extension options
adjust_species_mutation_magnitude_based_on_fitness = False
adjust_mutation_magnitudes_over_run = False
allow_elite_cloning = False

breed_mutagens = False
mutagen_breed_chance = 0.5


# --------------------------------------------------------------------------------------------------------------------#


# --------------------------------------------------------------------------------------------------------------------#
def get_device():
    """Used to obtain the correct device taking into account multiple GPUs"""

    gpu = 'cuda:'
    gpu += '0' if num_gpus <= 1 else mp.current_process().name
    return torch.device('cpu') if device == 'cpu' else torch.device(gpu)


def is_parallel():
    return not (device == 'cpu' or num_gpus <= 1)
