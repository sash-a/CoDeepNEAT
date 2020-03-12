import random

hyperbolic_alpha = 0.0024

from src.phenotype.neural_network.evaluator.training_results import hyperbolic_function, \
    get_current_gradient_given_training_data, linearise_data
import matplotlib.pyplot as plt


def get_fake_loss_values(noise_fac, alpha, epochs):
    losses = [hyperbolic_function(x, alpha, 0) + random.uniform(-noise_fac/2, noise_fac/2) for x in range(epochs)]
    return losses


def get_grad_line(x, y, gradient):
    c = y - gradient * x
    return [gradient * _x + c for _x in range(-20,100)]


def test_grad_detector(num_epochs):
    losses = get_fake_loss_values(0,hyperbolic_alpha, 100)
    gradient = get_current_gradient_given_training_data(losses[:num_epochs])
    epochs = list(range(100))
    linear_epochs = list(range(-20,100))
    plt.plot(epochs, losses)
    plt.plot(linear_epochs, get_grad_line(epochs[num_epochs], losses[num_epochs], gradient))
    plt.show()


def plot_gradient_over_epochs(num_epochs):
    # losses = get_fake_loss_values(0,hyperbolic_alpha, num_epochs)
    losses = read_prod_loss(1/167/256)
    print("loss at e5:",losses[4])
    epochs = list(range(1,num_epochs - 1))
    gradients = []
    for i in epochs:
        _losses = losses[:i+1]
        gradient = get_current_gradient_given_training_data(_losses)
        gradients.append(gradient)
    plt.plot(epochs, gradients)
    plt.xlabel("epochs")
    plt.ylabel("gradients")
    plt.show()


def read_prod_loss(scale=1.0):
    f = open("prod_loss_2.txt", "r")
    lines = f.readlines()
    epochs = []
    losses = []
    for line in lines:
        if "epoch" in line:
            line = line.split("epoch: ")[1]
            epochs.append(int(line))
        if "loss" in line:
            line = line.split("loss: ")[1]
            losses.append(float(line) * scale)

    return losses

if __name__ == "__main__":
    # test_grad_detector(3)
    # test_lineariser(60)
    plot_gradient_over_epochs(100)