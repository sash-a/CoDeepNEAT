import torch

# device = torch.device('cuda:0')
device = torch.device('cpu')

second_objective = "network_size"  # network_size|""
third_objective = ""

test_in_run = False
dummy_run = False
protect_parsing_from_errors = True

number_of_epochs_per_evaluation = 3

print_best_graphs = False
print_best_graph_every_n_generations = 5
print_failed_graphs = False
