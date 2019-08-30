import random

from src.Config import Config
from src.NeuralNetwork.ModuleNet import create_nn
from src.Validation import Evaluator, DataLoader
import copy
from src.DataAugmentation.AugmentationScheme import AugmentationScheme as AS
from src.NEAT.Mutagen import Mutagen
from src.NEAT.Mutagen import ValueType


def get_fully_trained_network(module_graph, data_augs, num_epochs = 300):
    train, test = DataLoader.load_data(dataset=module_graph.dataset)
    sample, _ = DataLoader.sample_data(Config.get_device(), dataset= module_graph.dataset)
    # module_graph.plot_tree_with_graphvis(title="before putting in model", file="before")

    module_graph_clone = copy.deepcopy(module_graph)
    model = create_nn(module_graph,sample, feature_multiplier= 1)
    # module_graph.plot_tree_with_graphvis(title="after putting in model", file = "after")
    print("training nn", model)
    Evaluator.print_epoch_every = 1

    da_phenotypes = [dagenome.to_phenotype() for dagenome in data_augs]


    print("fully training using augs:", data_augs)

    acc = Evaluator.evaluate(model, num_epochs, Config.get_device(), train_loader=train, test_loader=test,
                             print_accuracy=True, batch_size=256, augmentors= da_phenotypes )

    print("model trained on", num_epochs, "epochs scored:", acc)


    # model = create_nn(module_graph_clone, sample, feature_multiplier=0.8)
    # # module_graph.plot_tree_with_graphvis(title="after putting in model", file="after")
    # print("training nn", model)
    # acc = Evaluator.evaluate(model, num_epochs, Config.get_device(), train_loader=train, test_loader=test,
    #                          print_accuracy=True, batch_size=256)
    #
    # print("model trained on", num_epochs, "epochs scored:",acc)

def get_accuracy_for_network(model, da_scheme=None, batch_size=256):
    if Config.dummy_run:
        acc = random.random()
    else:
        acc = Evaluator.evaluate(model, Config.number_of_epochs_per_evaluation, Config.get_device(), batch_size,
                                 augmentors=[da_scheme])
    return acc
