from src.Config import Config

"""a convienient wrapper to contain all of an evolutionary runs scores for all its objectives"""

class GenerationData:
    def __init__(self, accuracies, generation_number, second_objective_values, third_objective_values):
        self.accuracies = accuracies
        self.generation_number = generation_number
        self.second_objective_values = second_objective_values
        self.third_objective_values = third_objective_values
        self.objectives = [accuracies, second_objective_values, third_objective_values]

    def get_average_of_objective(self, objective_no=0):
        objective = self.objectives[objective_no]
        if objective is None or len(objective) == 0:
            return None

        return sum(objective) / len(objective)

    def get_max_of_objective(self, objective_no=0):
        objective = self.objectives[objective_no]
        if objective is None or not objective:
            return None

        return max(objective)

    def get_top_n_average(self, n, objective=0):
        objectives = self.objectives[objective]
        if objectives is None or len(objectives) == 0:
            return None

        tops = []

        for score in objectives:
            i = 0
            if len(tops) == 0:
                tops.append(score)
                continue

            while i < len(tops) and i < n:
                if score > tops[i]:
                    tops.insert(i, score)
                    break
                i += 1

        tops = tops[:n]

        return sum(tops) / len(tops)

    def get_summary(self):
        """
            gen:num{obj~max:val;average:val,obj2..}
            comma separates different objectives
            ; separates max and average / other metrics
        """
        summary = "gen:" + repr(self.generation_number) + "{"
        for i in range(3):
            name = get_objective_name(i)
            average = self.get_average_of_objective(i)
            max = self.get_max_of_objective(i)
            if average is None:
                break
            if i > 0:
                summary += "|"
            summary += name + "~max:" + repr(max) + ";average:" + repr(average)

        summary += "}"
        return summary

    def get_data(self):
        data = "gen:" + repr(self.generation_number) + "{"
        for i in range(3):
            name = get_objective_name(i)
            values = self.objectives[i]
            if values is None or len(values) == 0:
                break
            datum = ("|" if i > 0 else " ") + name + ":" + repr(values)
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
