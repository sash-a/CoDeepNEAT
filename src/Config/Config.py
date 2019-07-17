import torch
import operator

# device = torch.device('cuda:0')
device = torch.device('cpu')
num_gpus = 4
num_workers = 0
dataset = 'mnist'

num_generations = 2

second_objective = "network_size"  # network_size
second_objective_comparator = operator.lt
third_objective = ""
third_objective_comparator = operator.lt

data_path = '../../data'

test_in_run = True
dummy_run = True
protect_parsing_from_errors = False

number_of_epochs_per_evaluation = 3

save_best_graphs = True
print_best_graphs = False
print_best_graph_every_n_generations = 2
save_failed_graphs = True
