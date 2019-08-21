import random

from src.Config import Config
from src.NeuralNetwork.ModuleNet import create_nn
from src.Validation import Evaluator, DataLoader
import copy
from src.DataAugmentation.AugmentationScheme import AugmentationScheme as AS
from src.NEAT.Mutagen import Mutagen
from src.NEAT.Mutagen import ValueType


def get_fully_trained_network(module_graph, data_augs, num_epochs = 15):
    train, test = DataLoader.load_data(dataset=module_graph.dataset)
    sample, _ = DataLoader.sample_data(Config.get_device(), dataset= module_graph.dataset)
    # module_graph.plot_tree_with_graphvis(title="before putting in model", file="before")

    module_graph_clone = copy.deepcopy(module_graph)
    model = create_nn(module_graph,sample, feature_multiplier= 0.8)
    # module_graph.plot_tree_with_graphvis(title="after putting in model", file = "after")
    print("training nn", model)
    Evaluator.print_epoch_every = 1

    da_phenotypes = [dagenome.to_phenotype() for dagenome in data_augs]

    # augSc = AS(None, None)
    # da_submutagens = {
    #     "Grayscale": {
    #         "alpha_lo": Mutagen(value_type=ValueType.CONTINUOUS, current_value=0.35, start_range=0.0,
    #                             end_range=0.49, mutation_chance=0.3),
    #         "alpha_hi": Mutagen(value_type=ValueType.CONTINUOUS, current_value=0.75, start_range=0.5,
    #                             end_range=1.0, mutation_chance=0.3)}
    # }
    # augSc.add_augmentation(Mutagen("Grayscale", sub_mutagens=da_submutagens))
    # augSc.add_augmentation(Mutagen("Flip_lr"))
    # da_phenotypes = [augSc]

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
