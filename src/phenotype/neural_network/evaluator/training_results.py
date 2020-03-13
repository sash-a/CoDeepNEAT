# from scipy.optimize import curve_fit
import numpy as np


class TrainingResults:
    """
        an object to store the loss values and intermediate accuracy readings of a training session
    """

    def __init__(self):
        self.losses = []  # it is assumed a loss value is provided every epoch
        self.accuracies = []
        self.accuracy_epochs = []  # the epochs the accs were read at

    def add_loss(self, loss):
        self.losses.append(loss)

    def add_accuracy(self, acc, epoch):
        self.accuracies.append(acc)
        self.accuracy_epochs.append(epoch)

    def get_loss_gradient(self):
        return get_current_gradient_given_training_data(self.losses)

    def get_max_acc_age(self):
        max_acc = -1
        max_acc_age = -1  # number of worse accuracies after the max acc

        for acc in self.accuracies:
            if acc > max_acc:
                max_acc = acc
                max_acc_age = 0
            else:
                max_acc_age += 1

        return max_acc_age

    def get_max_acc(self):
        if len(self.accuracies) == 0:
            return -1
        return max(self.accuracies)


def get_current_gradient_given_training_data(y_data, x_data=None):
    if x_data is None:
        x_data = list(range(len(y_data)))  # if not specified - the x data is just 0..n
    # hyperbolic_params, _ = curve_fit(hyperbolic_function, x_data, y_data)
    linearised_data = linearise_data(y_data)
    try:
        params = np.polyfit(x_data, linearised_data, 2)
    except Exception as e:
        print("error fitting linearised data - ",linearised_data)
        raise e
    m = params[1]

    a = 1/m
    current_gradient = derivative_hyperbolic_function(x_data[-1], a)
    print("e:",len(x_data),"gradient value:", current_gradient)
    return current_gradient


def linearise_data(y_data):
    """transforms the hyperbolic data into linear"""
    return [1/y for y in y_data]


# the hyperbolic function used is shifted 1E to the left
# this is because at e=0, the loss has a real value already
def derivative_hyperbolic_function(x, a):
    return - pow(x + 1, -2) * a


def hyperbolic_function(x, a, q):
    return q + a / (x + 1)

