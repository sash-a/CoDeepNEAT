import math
import sys
import matplotlib.pyplot as plt

def get_data_efficiency(tuples):
    """
    :param tuples: list of (training_set_fraction, accuracy) tuples
    :return: the data efficiency value for the given (training_set_fraction, accuracy) tuples
    """
    _,max_acc = tuples[-1]
    return math.pow(solve_for_learning_rate(tuples), 0.1) * max_acc

def solve_for_learning_rate(tuples):
    """
    adaptively tries to minimise MSE wrt learning rate
    :param tuples: a list of (training_set_fraction, accuracy) tuples to fit
    :return: the closest learning rate which could be found, which best fits the DE curve over the given tuples
    """

    learning_rate = 0.000001

    min_MSE = sys.maxsize
    best_learning_rate = -1

    change_frac = 2#decreasing change frac means more sensitive updating of learning rate

    previous_MSE = min_MSE
    dir = 1 #either 1 or -1. 1 for increase lr, -1 for decrease lr
    for i in range(100000):
        MSE = get_mean_squared_error(tuples, learning_rate)
        if MSE<min_MSE:
            min_MSE = MSE
            best_learning_rate = learning_rate
            #print("new best learning rate:",best_learning_rate, "MSE:",MSE)

        if MSE > previous_MSE:
            """gotten worse - change dir, decrease change_frac"""
            dir*=-1
            change_frac = math.pow(change_frac,0.85)

        learning_rate*= math.pow(change_frac,dir)
        previous_MSE = MSE
    return best_learning_rate


def get_mean_squared_error(tuples, learning_rate):
    _,max_acc = tuples[-1]
    squared_error = 0
    for set_fraction, acc in tuples:
        predicted_acc = get_predicted_accuracy(max_acc, set_fraction, learning_rate)
        squared_error += math.pow(predicted_acc-acc, 2)
    return squared_error

def get_predicted_accuracy(max_accuracy, training_set_fraction, learning_rate):
    A = 1/learning_rate
    return max_accuracy * math.log((training_set_fraction + A)/A) * math.pow(A,get_fix_number(learning_rate))

def get_fix_number(learning_rate):
    A = 1/learning_rate
    if A == 1:
        A = 1.01
    try:
        return math.log(0.434294482/math.log((100+A)/A, 10), 10)/math.log(A,10)
    except:
        raise Exception("error getting fix number at lr=",learning_rate,"A=",A)

def plot_tuples_with_best_fit(tuples, learning_rate = None):
    if learning_rate is None:
        learning_rate = solve_for_learning_rate(tuples)
    _,max_acc = tuples[-1]


    plt.plot([list(x)[0] for x in tuples], [list(y)[1] for y in tuples], label = "data")
    plt.plot([x for x in range(100)], [get_predicted_accuracy(max_acc, x, learning_rate) for x in range(100)], label = "best fit" )
    plt.title("best fit")
    plt.xlabel("% of full training set")
    plt.ylabel("% classification accuracy")
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.gca().legend(handles, labels)
    plt.xlim([0, 100])
    plt.ylim([0, 100])
    plt.show()

tuples =[(10,51.5),(20,58),(30,58.5),(41,61.5),(50,64),(61,65),(71,64),(81,66),(91,67),(100,66)]
print("lr:", solve_for_learning_rate(tuples), "DE:", get_data_efficiency(tuples), "max:",tuples[-1])

tuples = [(10,29.5),(20,40.5),(30,46),(41,51),(51,53),(61,54),(71,56),(81,57.5),(91,59),(100,61)]
print("lr:", solve_for_learning_rate(tuples), "DE:", get_data_efficiency(tuples), "max:",tuples[-1])


