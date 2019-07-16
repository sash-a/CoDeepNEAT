device = 'cuda:0'
# device = torch.device('cpu')
num_generations = 500

second_objective = "network_size"  # network_size
third_objective = ""

data_path = '../../data'

test_in_run = True
dummy_run = False
protect_parsing_from_errors = False

number_of_epochs_per_evaluation = 3

save_best_graphs = True
print_best_graphs = False
print_best_graph_every_n_generations = 2
save_failed_graphs = True
