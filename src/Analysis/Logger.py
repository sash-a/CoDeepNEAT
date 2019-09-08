import ast
import os
import os.path
from typing import List
from data import DataManager
from src.Analysis.GenerationData import GenerationData

"""the logging tool for saving/reading the results of runs during evolution"""

generations: List[GenerationData] = []

def log_new_generation(accuracies, generation_number, second_objective_values=None, third_objective_values=None,
                       write_summaries=False):
    global generations
    global log_file

    generations.append(GenerationData(accuracies, generation_number, second_objective_values, third_objective_values))

    if not os.path.exists(DataManager.get_Logs_folder()):
        os.makedirs(DataManager.get_Logs_folder())

    with open(os.path.join(DataManager.get_Logs_folder(), log_file), "a+") as f:
        if write_summaries:
            f.write(generations[-1].get_summary() + "\n")
        else:
            f.write(generations[-1].get_data() + "\n")


def load_date_from_log_file(run_name, iteration=0, summary=False):
    global generations
    generations = []
    filename = os.path.join(DataManager.get_Logs_folder(run_name),
                            "log" + (("_" + repr(iteration)) if iteration > 0 else ""))
    log = open(filename + ".txt")
    for gen in log:
        gen_number = int(gen.split("{")[0].split(":")[1])
        gen = gen.split("{")[1].split("}")[0]
        objectives = gen.split("|")
        o = 0
        accuracies = second = third = None
        for objective in objectives:
            if summary:
                name = objective.split("~")[0]
                vals = objective.split("~")[1].split(";")

                max = None
                av = None
                for val in vals:
                    if "max" in val.split(":")[0]:
                        max = float(val.split(":")[1])
                    if "average" in val.split(":")[0]:
                        av = float(val.split(":")[1])
            else:
                name = objective.split(":")[0]
                vals = objective.split(":")[1]
                if "accuracy" in name:
                    if o > 0:
                        print("warning, accuracy not the first objective in log", filename)
                        return
                    accuracies = ast.literal_eval(vals)
                elif o == 1:
                    second = ast.literal_eval(vals)
                elif o == 2:
                    third = ast.literal_eval(vals)
                else:
                    raise Exception("too many objectives in log", filename, o, name)
            o += 1
        generations.append(GenerationData(accuracies, gen_number, second, third))


def get_next_log_file_name(log_file_name=None):
    if log_file_name is None:
        log_file_name = "log"

    file_exists_already = os.path.isfile(os.path.join(DataManager.get_Logs_folder(), log_file_name + ".txt"))
    if file_exists_already:
        counter = 1
        while file_exists_already:
            file_exists_already = os.path.isfile(
                os.path.join(DataManager.get_Logs_folder(), log_file_name + "_" + repr(counter) + ".txt"))
            counter += 1
        counter -= 1
        log_file_name = log_file_name + "_" + repr(counter)
    return log_file_name + ".txt"


log_file = get_next_log_file_name()
