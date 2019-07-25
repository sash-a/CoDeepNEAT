from src.Config import Config
from src.Validation import Evaluator
from src.NeuralNetwork.ModuleNet import create_nn

def cross_validate(module_graph, dataset="" ,k = 10):
    model = create_nn(module_graph, )

def validate_fold(model, dataset, k, i):
    pass


def get_accuracy_for_network(model, da_scheme=None, batch_size=256):
    if Config.dummy_run:
        acc = hash(model)
    else:
        acc = Evaluator.evaluate(model, Config.number_of_epochs_per_evaluation, Config.get_device(), batch_size,
                                 augmentor=da_scheme)
    return acc
