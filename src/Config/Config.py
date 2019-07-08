import torch

device=torch.device('cuda:0')
#device=torch.device('cpu')

dummy_run = True
protect_parsing_from_errors = False
print_graphs=False