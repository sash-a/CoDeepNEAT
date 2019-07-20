import torch
import operator
import multiprocessing as mp

run_name = "test_run"

# device = 'gpu'
device = 'cpu'
num_gpus = 1
num_workers = 0
dataset = 'mnist'

num_generations = 20

second_objective = "network_size"  # network_size
second_objective_comparator = operator.lt  # lt for minimisation, gt for maximisation
third_objective = ""
third_objective_comparator = operator.lt

moo_optimiser = "cdn"  # cdn/nsga

data_path = ''

test_in_run = False
dummy_run = True
protect_parsing_from_errors = True

number_of_epochs_per_evaluation = 1

save_best_graphs = True
print_best_graphs = False
print_best_graph_every_n_generations = 5
save_failed_graphs = True


def get_device():
    """Used to obtain the correct device taking into account multiple gpus"""
    gpu = 'cuda:'
    gpu += '0' if num_gpus <= 1 else str(int(mp.current_process().name[-1]) % num_gpus)

    return torch.device('cpu') if device == 'cpu' else torch.device(gpu)


def is_parallel():
    return not (device == 'cpu' or num_gpus <= 1)
