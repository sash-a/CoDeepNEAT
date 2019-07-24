import math

def solve_for_learning_rate(tuples):
    pass

def get_mean_squared_error(tuples, learning_rate):
    _,max_acc = tuples[-1]

def get_predicted_accuracy(max_accuracy, training_set_fraction, learning_rate):
    A = 1/learning_rate
    return max_accuracy * math.log((training_set_fraction + A)/A) * math.pow(A,get_fix_number(learning_rate))

def get_fix_number(learning_rate):
    A = 1/learning_rate
    return math.log(0.434294482/math.log((100+A)/A, 10), 10)/math.log(A,10)

