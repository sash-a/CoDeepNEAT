import sys
from src.Config import Config

class GenerationData:

    def __init__(self, accuracies, generation_number,second_objective_values,third_objective_values):
        self.accuracies = accuracies
        self.generation_number = generation_number
        self.second_objective_values = second_objective_values
        self.third_objective_values = third_objective_values
        self.objectives = [accuracies, second_objective_values, third_objective_values]

    def get_average_of_objective(self, objective = 0):
        objectives = self.objectives[objective]
        if objectives is None or len(objectives) == 0:
            return None
        total = 0
        for val in objectives:
            total+= val

        return total/len(objectives)


    def get_max__of_objective(self, objective = 0):
        objectives = self.objectives[objective]
        if objectives is None or len(objectives) == 0:
            return None

        max_val = -sys.maxsize -1
        for val in self.accuracies:
            max_val = max(val,max_val)
        return max_val

    def get_summary(self):
        """gen:num{obj~max:val;average:val,obj2..}
            comma separates different objectives
            ; separates max and average / other metrics
        """
        summary = "gen:"+ repr(self.generation_number)+"{"
        for i in range(3):
            name = get_objective_name(i)
            average = self.get_average_of_objective(i)
            max = self.get_max__of_objective(i)
            if average is None:
                break
            if i>0:
                summary+="|"
            summary+=name+"~max:"+repr(max)+";average:"+repr(average)

        summary+="}"
        return summary

    def get_data(self):
        data = "gen:"+ repr(self.generation_number)+"{"
        for i in range(3):
            name = get_objective_name(i)
            values = self.objectives[i]
            if values is None or len(values) == 0:
                break
            datum = ("|" if i>0 else " ") + name+":" + repr(values)
            print(datum)
            data += datum

        data += "}"
        return data


def get_objective_name(objective_number):
    if objective_number == 0:
        return "accuracy"
    if objective_number == 1:
        return Config.second_objective
    if objective_number == 2:
        return Config.third_objective
