import copy
import random

from src.Config import Config
from src.NeuralNetwork.ModuleNet import create_nn
from src.Validation import Evaluator, DataLoader


def get_fully_trained_network(module_graph, data_augs, num_epochs=100, plot_best_graph=False):
    """takes a module graph, converts it into a module net NN and trains it for n epochs.
    used to fully train a saved module graph to find its max accuracy
    """

    train, test = DataLoader.load_data(dataset=module_graph.dataset)
    sample, _ = DataLoader.sample_data(Config.get_device(), dataset=module_graph.dataset)

    module_graph_backup = copy.deepcopy(module_graph)

    model = create_nn(module_graph, sample, feature_multiplier=Config.feature_multiplier_for_fully_train)
    if plot_best_graph:
        module_graph.plot_tree_with_graphvis(title="after putting in model", file="after")

    print("fully training ~ num epochs:", num_epochs, "num augs:", len(data_augs), "feature multiplier:",
          Config.feature_multiplier_for_fully_train)
    Evaluator.print_epoch_every = 1

    da_phenotypes = [dagenome.to_phenotype() for dagenome in data_augs]

    if len(data_augs) > 0:
        print("fully training using augs:", data_augs)

    acc = Evaluator.evaluate(model, num_epochs, Config.get_device(), train_loader=train, test_loader=test,
                             print_accuracy=True, batch_size=Config.batch_size, augmentors=da_phenotypes,
                             training_target=module_graph.fitness_values[0])

    if Config.toss_bad_runs and acc == "toss":
        print("runn failed to take off. starting over")
        get_fully_trained_network(module_graph_backup, data_augs, num_epochs=num_epochs,
                                  plot_best_graph=plot_best_graph)

    print("model trained on", num_epochs, "epochs scored:", acc)


def get_accuracy_estimate_for_network(model, da_scheme=None, batch_size=Config.batch_size):
    """used in the evolutionary process to  train a model for the number of epochs specified in Config settings"""
    if Config.dummy_run:
        acc = random.random()
    else:
        acc = Evaluator.evaluate(model, Config.number_of_epochs_per_evaluation, Config.get_device(), batch_size,
                                 augmentors=[da_scheme])
    return acc
