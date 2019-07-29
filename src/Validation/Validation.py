from src.Config import Config
from src.Validation import Evaluator, DataLoader
from src.Validation.DataSet import DataSet
from src.NeuralNetwork.ModuleNet import create_nn
from data import DataManager
#from sklearn.model_selection import cross_val_predict


def cross_validation(run_name):
    gen_state = DataManager.load_generation_state(run_name)
    best_graph = gen_state.pareto_population.get_highest_accuracy()
    train_loader, test_loader = DataLoader.load_data()
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        y_pred = cross_val_predict(best_graph, inputs, targets, cv=5)
        print(y_pred)


# def cross_validate(module_graph, dataset="", k=10):
#     sample_train, sample_test = DataLoader.sample_data(Config.get_device(), dataset=dataset)
#     dataset = DataSet(sample_train, sample_test)
#     if dataset is None:
#         raise Exception("null dataset")
#     total_acc = 0
#     for i in range(k):
#         model = create_nn(module_graph, sample_train)
#         acc = validate_fold(model, None, dataset, k, i)
#         # acc = validate_fold(model, dataset, module_graph.data_augmentation_schemes[0], k, i)
#
#         total_acc += acc
#         module_graph.delete_all_layers()
#         print(k, "fold validation", i, ":", acc)
#
#     return total_acc / k
#
#
# def validate_fold(model, da_scheme, dataset, k, i):
#     if dataset is None:
#         raise Exception("null dataset")
#
#     acc = Evaluator.evaluate(model, 2, Config.get_device(), batch_size=1,
#                              augmentor=da_scheme, train_loader=dataset.get_training_fold(k, i),
#                              test_loader=dataset.get_testing_fold(k, i))
#
#     return acc
#
#
def get_accuracy_for_network(model, da_scheme=None, batch_size=256):
    if Config.dummy_run:
        acc = hash(model)
    else:
        acc = Evaluator.evaluate(model, Config.number_of_epochs_per_evaluation, Config.get_device(), batch_size,
                                 augmentor=da_scheme)
    return acc
