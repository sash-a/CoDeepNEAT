from src.Config.Config import Config
from src.NeuralNetwork.ModuleNet import create_nn
from src.Validation import Evaluator, DataLoader


# from sklearn.model_selection import cross_val_predict
# from skorch import NeuralNetClassifier


# def cross_validation(run_name):
#     gen_state = DataManager.load_generation_state(run_name)
#     best_graph = gen_state.pareto_population.get_highest_accuracy()
#     train_loader, test_loader = DataLoader.load_data()
#     net = NeuralNetClassifier(module=best_graph, train_split=None)
#     x_train_loader, y_train_loader = [], []
#     for batch_idx, (inputs, targets) in enumerate(train_loader):
#         x_train_loader.append(inputs)
#         y_train_loader.append(targets)
#
#     np_x = np.asarray(x_train_loader)
#     ts_x = torch.from_numpy(np_x)
#     np_y = np.asarray(y_train_loader)
#     ts_y = torch.from_numpy(np_y)
#
#     y_pred = cross_val_predict(net, ts_x, ts_y, cv=5)


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

def get_fully_trained_network(module_graph, data_augs, num_epochs=30):
    train, test = DataLoader.load_data(dataset=module_graph.dataset)
    sample, _ = DataLoader.sample_data(Config.get_device(), dataset=module_graph.dataset)
    module_graph.plot_tree_with_graphvis(title="before putting in model")
    model = create_nn(module_graph, sample)
    module_graph.plot_tree_with_graphvis(title="after putting in model")
    acc = Evaluator.evaluate(model, num_epochs, Config.get_device(), train_loader=train, test_loader=test)

    print("model trained on", num_epochs, "epochs scored:", acc)


def get_accuracy_for_network(model, da_scheme=None, batch_size=256):
    if Config.dummy_run:
        acc = hash(model)
    else:
        acc = Evaluator.evaluate(model, Config.number_of_epochs_per_evaluation, Config.get_device(), batch_size,
                                 augmentors=[da_scheme])
    return acc
