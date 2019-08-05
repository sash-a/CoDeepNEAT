import torch
import operator
import torch.multiprocessing as mp

# --------------------------------------------------------------------------------------------------------------------#
# Run state options
run_name = "test"
continue_from_last_run = False
fully_train = False
# --------------------------------------------------------------------------------------------------------------------#
# nn options
device = 'gpu'  # gpu | cpu
num_gpus = 4
num_workers = 0  # this doesn't work in parallel because daemonic processes cannot spawn children
dataset = 'cifar10'
data_path = ''
number_of_epochs_per_evaluation = 5

# --------------------------------------------------------------------------------------------------------------------#
max_num_generations = 25

# --------------------------------------------------------------------------------------------------------------------#
# Multiobjective options
second_objective = 'network_size'  # network_size | network_size_adjusted | network_size_adjusted_2
second_objective_comparator = operator.lt  # lt for minimisation, gt for maximisation
third_objective = ''
third_objective_comparator = operator.lt

moo_optimiser = 'cdn'  # cdn | nsga

# --------------------------------------------------------------------------------------------------------------------#
# Data augmentation options
evolve_data_augmentations = False
colour_augmentations = False

# --------------------------------------------------------------------------------------------------------------------#
module_retention = False
# --------------------------------------------------------------------------------------------------------------------#
speciation_overhaul = False
ignore_disabled_connections_for_topological_similarity = False
# --------------------------------------------------------------------------------------------------------------------#
use_graph_edit_distance = False
# --------------------------------------------------------------------------------------------------------------------#
fitness_aggregation = 'avg'  # max | avg
# --------------------------------------------------------------------------------------------------------------------#

# Debug and test options
dummy_run = False

protect_parsing_from_errors = False
test_in_run = False
interleaving_check = False

save_best_graphs = False
print_best_graphs = False
print_best_graph_every_n_generations = 5
save_failed_graphs = False


# --------------------------------------------------------------------------------------------------------------------#
def get_device():
    """Used to obtain the correct device taking into account multiple GPUs"""

    gpu = 'cuda:'
    gpu += '0' if num_gpus <= 1 else mp.current_process().name
    return torch.device('cpu') if device == 'cpu' else torch.device(gpu)


def is_parallel():
    return not (device == 'cpu' or num_gpus <= 1)
