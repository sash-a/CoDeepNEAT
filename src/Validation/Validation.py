from src.Config import Config
from src.Validation import Evaluator,DataLoader
from src.Validation.DataSet import DataSet
from src.NeuralNetwork.ModuleNet import create_nn


def cross_validate(module_graph, dataset="" ,k = 10):
    sample_train,sample_test = DataLoader.sample_data(Config.get_device(), dataset=dataset)
    dataset = DataSet(sample_train,sample_test)
    total_acc=0
    for i in range(k):
        model = create_nn(module_graph, sample_train)
        acc = validate_fold(model,dataset, module_graph.data_augmentation_schemes[0],k,i)
        total_acc += acc
        module_graph.delete_all_layers()
        print(k,"fold validation",i,":",acc)

    return acc/10

def validate_fold(model, da_scheme,dataset, k, i):

    acc = Evaluator.evaluate(model, Config.number_of_epochs_per_evaluation, Config.get_device(), batch_size=1,
                             augmentor=da_scheme, train_loader= dataset.get_training_fold(k,i), test_loader=dataset.get_testing_fold(k,i))

    return acc


def get_accuracy_for_network(model, da_scheme=None, batch_size=256):
    if Config.dummy_run:
        acc = hash(model)
    else:
        acc = Evaluator.evaluate(model, Config.number_of_epochs_per_evaluation, Config.get_device(), batch_size,
                                 augmentor=da_scheme)
    return acc
