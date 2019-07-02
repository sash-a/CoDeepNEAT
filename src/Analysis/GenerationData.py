import sys

class GenerationData:

    def __init__(self, accuracies, generation_number):
        self.accuracies = accuracies
        self.generation_number = generation_number

    def average_accuracy(self):
        total_accuracy = 0
        for acc in self.accuracies:
            total_accuracy += acc
        return total_accuracy/len(self.accuracies)

    def max_accuracy(self):
        max = -sys.maxsize -1
        for acc in self.accuracies:
            if(acc > max):
                max = acc
        return max

    def get_summary(self):
        return "gen: " + repr(self.generation_number)+" accuracy~ max: "+ repr(self.max_accuracy()) + " ; average: " + repr(self.average_accuracy())