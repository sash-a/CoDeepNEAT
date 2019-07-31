import torch
import operator
import torch.multiprocessing as mp

# --------------------------------------------------------------------------------------------------------------------#
# Run state options
run_name = "test_run"
continue_from_last_run = True

# --------------------------------------------------------------------------------------------------------------------#
# nn options
device = 'cpu'  # gpu | cpu
num_gpus = 1
num_workers = 0  # this doesn't work in parallel because daemonic processes cannot spawn children
dataset = 'fashion_mnist'
data_path = ''
number_of_epochs_per_evaluation = 5

# --------------------------------------------------------------------------------------------------------------------#
max_num_generations = 150

# --------------------------------------------------------------------------------------------------------------------#
# Multiobjective options
second_objective = 'network_size_adjusted_2'  # network_size | network_size_adjusted | network_size_adjusted_2
second_objective_comparator = operator.lt  # lt for minimisation, gt for maximisation
third_objective = ''
third_objective_comparator = operator.lt

moo_optimiser = 'cdn'  # cdn | nsga

# --------------------------------------------------------------------------------------------------------------------#
# Data augmentation options
evolve_data_augmentations = True
colour_augmentations = True

# --------------------------------------------------------------------------------------------------------------------#
maintain_module_handles = False

# --------------------------------------------------------------------------------------------------------------------#
# Debug and test options
dummy_run = True

protect_parsing_from_errors = False
test_in_run = False
interleaving_check = False

save_best_graphs = True
print_best_graphs = True
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
