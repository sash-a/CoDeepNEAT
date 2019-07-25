import torch
import operator
import torch.multiprocessing as mp

# --------------------------------------------------------------------------------------------------------------------#
# Run state options
run_name = "test_run"
continue_from_last_run = False

# --------------------------------------------------------------------------------------------------------------------#
# nn options
device = 'gpu'
# device = 'cpu'
num_gpus = 1
num_workers = 0  # this doesn't work in parallel because daemonic processes cannot spawn children
dataset = 'fassion_mnist'
data_path = ''
number_of_epochs_per_evaluation = 5

# --------------------------------------------------------------------------------------------------------------------#
max_num_generations = 100

# --------------------------------------------------------------------------------------------------------------------#
# Multiobjective options
second_objective = "network_size"  # network_size
second_objective_comparator = operator.lt  # lt for minimisation, gt for maximisation
third_objective = ""
third_objective_comparator = operator.lt

moo_optimiser = "cdn"  # cdn/nsga

# --------------------------------------------------------------------------------------------------------------------#
# Data augmentation options
evolve_data_augmentations = True

# --------------------------------------------------------------------------------------------------------------------#
# Debug and test options
dummy_run = True

protect_parsing_from_errors = False
test_in_run = False
interleaving_check = False

save_best_graphs = True
print_best_graphs = False
print_best_graph_every_n_generations = 5
save_failed_graphs = True


# --------------------------------------------------------------------------------------------------------------------#

def get_device():
    """Used to obtain the correct device taking into account multiple gpus"""

    gpu = 'cuda:'
    gpu += '0' if num_gpus <= 1 else mp.current_process().name
    # return torch.device('cuda:0')
    return torch.device('cpu') if device == 'cpu' else torch.device(gpu)


def is_parallel():
    return not (device == 'cpu' or num_gpus <= 1)
