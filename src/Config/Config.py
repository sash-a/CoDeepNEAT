import torch
import operator
import torch.multiprocessing as mp

run_name = "test_run"
continue_from_last_run = False



device = 'gpu'
#device = 'cpu'
num_gpus = 1
num_workers = 2
dataset = 'mnist'
data_path = ''



max_num_generations = 150
number_of_epochs_per_evaluation = 3



second_objective = "network_size"  # network_size
second_objective_comparator = operator.lt  # lt for minimisation, gt for maximisation
third_objective = ""
third_objective_comparator = operator.lt

moo_optimiser = "cdn"  # cdn/nsga



evolve_data_augmentations = True



test_in_run = False
dummy_run = True
protect_parsing_from_errors = False
save_best_graphs = True
print_best_graphs = False
print_best_graph_every_n_generations = 5
save_failed_graphs = True

interleaving_check = True


def get_device():
    """Used to obtain the correct device taking into account multiple gpus"""

    gpu = 'cuda:'
    gpu += '0' if num_gpus <= 1 else str(int(mp.current_process().pid % num_gpus))
    # return torch.device('cuda:0')
    return torch.device('cpu') if device == 'cpu' else torch.device(gpu)


def is_parallel():
    return not (device == 'cpu' or num_gpus <= 1)
