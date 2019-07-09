import torch

device = torch.device('cuda:0')
# device = torch.device('cpu')

test_in_run = False
dummy_run = True
protect_parsing_from_errors = True
print_best_graphs = False
print_failed_graphs = True
