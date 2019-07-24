import torch
import torchvision.transforms as transforms
from src.DataEfficiency import Net
from src.NeuralNetwork.Evaluator import load_data
import math
import matplotlib.pyplot as plt

import torch.nn as nn
import torch.optim as optim

criterion = nn.CrossEntropyLoss()
#networks = [ Net.DropOutNet ,Net.StandardNet,Net.BatchNormNet]
networks = [ Net.DropOutNet ,Net.DropOutNet ,Net.DropOutNet,Net.DropOutNet,Net.DropOutNet,Net.DropOutNet,Net.DropOutNet   ]
total_batches = None

def test_model(model):
    model.eval()
    correct = 0
    total = 0

    trainloader, testloader = load_data(dataset="cifar10")

    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(torch.device("cuda:0")), labels.to(torch.device("cuda:0"))

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    #print('Accuracy of the network on the 10000 test images: %d %%' % (accuracy))

    return accuracy

def run_epoch_for_n_batches(model,optimiser, num_batches = -1):
    trainloader, testloader = load_data(dataset="cifar10")

    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        if i+2 >= num_batches > 0:
            return
        #print("training",model)
        inputs, labels = data
        inputs,labels = inputs.to(torch.device("cuda:0")), labels.to(torch.device("cuda:0"))

        # zero the parameter gradients
        optimiser.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimiser.step()




def run_model_over_different_batch_numbers(num_epochs, model_type):
    num_batches = 1
    accuracies = []#tuples of (%training_set, %accuracy)
    for i in range(11):

        model = model_type().to(torch.device("cuda:0"))
        optimiser = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        for epoch in range(num_epochs):  # loop over the dataset multiple times

            run_epoch_for_n_batches(model,optimiser,num_batches=num_batches)

        accuracy = test_model(model)
        training_proportion = 100*num_batches/total_batches

        print("num_batches:",num_batches,"/",total_batches,("("+str(round(training_proportion))+"%)"),"acc:",accuracy,"%")


        num_batches += math.ceil(total_batches/10)
        num_batches = min(total_batches,num_batches)

        accuracies.append((training_proportion,accuracy))


    return accuracies

def test_all_networks(num_epochs):
    plot_points = []

    for network_type in networks:
        print(get_name_from_class(network_type))
        accuracies = run_model_over_different_batch_numbers(num_epochs,network_type)
        #plot_model_accuracies(accuracies, network_type)
        plot_points.append((accuracies,network_type))

    plot_all_accuracies(plot_points)



def test_max_accuracy_of_networks(num_epochs):
    i=0
    for network_type in networks:
        dropout = 0.03 + 0.02 * math.pow(i, 2)
        print("dropout:",dropout)
        model = network_type(drop_out_factor = dropout).to(torch.device("cuda:0"))
        optimiser = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        for epoch in range(num_epochs):  # loop over the dataset multiple times

            run_epoch_for_n_batches(model, optimiser, num_batches=total_batches)

        accuracy = test_model(model)
        print(get_name_from_class(network_type),"max acc:",accuracy)
        i+=1


def plot_model_accuracies(accuracies, model_type):
    plt.plot([list(x)[0] for x in accuracies], [list(x)[1] for x in accuracies])
    plt.title(get_name_from_class(model_type))
    plt.xlabel("% of full training set")
    plt.ylabel("% classification accuracy")
    plt.xlim([0,100])
    plt.ylim([0,100])
    plt.show()

def plot_all_accuracies(values):
    for model_values in values:
        accuracies, model_type = model_values
        name = get_name_from_class(model_type)
        plt.plot([list(x)[0] for x in accuracies], [list(x)[1] for x in accuracies], label = name)
        plt.xlabel("% of full training set")
        plt.ylabel("% classification accuracy")
    plt.xlim([0, 100])
    plt.ylim([0, 100])
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.gca().legend(handles,labels)
    plt.show()


def get_name_from_class(model_class):
    return repr(model_class).split(".")[-1].split("'")[0]


def run_tests():
    num_epochs = 20
    global total_batches
    trainloader, testloader = load_data(dataset="cifar10")
    total_batches = len(trainloader)
    test_max_accuracy_of_networks(num_epochs)
    test_all_networks(num_epochs)

if __name__ == "__main__":
    run_tests()
