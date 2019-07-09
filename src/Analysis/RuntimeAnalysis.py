from src.Analysis.GenerationData import GenerationData
import inspect, os
import os.path

generations = []
log_file = None


def log_new_generation(accuracies, generation_number, second_objective_values = None, third_objective_values = None):
    global generations
    global log_file

    if generation_number is None:
        generation_number = len(generations)

    generations.append(GenerationData(accuracies, generation_number,second_objective_values,third_objective_values))

    with open(get_log_folder() + log_file, "a+") as f:
        f.write(generations[-1].get_summary() + "\n")


def get_next_log_file_name(log_file_name=None):
    if log_file_name is None:
        log_file_name = "log"

    file_exists_already = os.path.isfile(get_log_folder() + log_file_name + ".txt")
    if (file_exists_already):
        counter = 1
        while (file_exists_already):
            file_exists_already = os.path.isfile(get_log_folder() + log_file_name + "_" + repr(counter) + ".txt")
            counter += 1
        counter -= 1
        log_file_name = log_file_name + "_" + repr(counter)
    return log_file_name + ".txt"


def configure(log_file_name=None):
    global log_file
    log_file = get_next_log_file_name(log_file_name)


def get_log_folder():
    return os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) + "\\Logs\\"
