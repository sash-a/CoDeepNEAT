import os

import matplotlib.pyplot as plt
from data import DataManager
from src.Analysis.EvolutionaryDataPlotter import name_overrides

"""reads, parses and reports on the maximum accuracies attained by each of the netwokrs 
which were fully trained, and had their results put into the results folder"""

def get_accuracies(results_lines):
    accs = []
    for line in results_lines:
        try:

            if "accuracy:" in line:
                """results file was copied from out file"""
                acc = line.split("i")[0].split("accuracy:")[1].replace(" ", "")
                accs.append(float(acc))
            elif "loss" in line:
                """results file itself was taken"""
                acc = line.split(",")[0].split(":")[1].replace(" ", "")
                accs.append(float(acc))
        except:
            print(line)

    return accs


def get_all_results_folders(dataset):
    folders = set()
    # print("getting results from:",os.path.join(DataManager.get_results_folder(), dataset,"fully_train"))
    for subdir, dirs, files in os.walk(os.path.join(DataManager.get_results_folder(), dataset,"fully_train")):
        sub = subdir.split("fully_train")[1][1:].split("\\")[0].split("/")[0]
        if sub == "":
            continue
        # print("result folder:",sub)
        folders.add(sub)

    return folders


def get_all_results_files_in_folder(dataset,folder):
    files = set()
    for subdir, dirs, files in os.walk(os.path.join(DataManager.get_results_folder(), dataset,"fully_train",folder)):
        sub = subdir.split(folder)[1][1:].split("\\")[0].split("/")[0]
        # print(sub)
        if sub == "":
            continue
        files.add(sub)

    return files


def print_max_accuracies(dataset):
    for run in get_all_results_folders(dataset):
        run_name = run
        if run_name in name_overrides:
            run_name = name_overrides[run_name]
        print(run_name)

        for result_file in get_all_results_files_in_folder(dataset,run):
            file_path = os.path.join(DataManager.get_results_folder(), dataset,"fully_train",run, result_file)
            # print(result_file,file_path)
            with open(file_path) as file:
                lines = file.readlines()
                accuracies = get_accuracies(lines)
                # print(accuracies)
                max_acc = max(accuracies)
                config_name = result_file.replace(".txt", "")
                print("\t", config_name, "max acc:", max_acc)


def get_fm_acc_tuples(dataset):
    data = {}  # dict from run: {config:(fm,acc)}

    for run in get_all_results_folders(dataset):
        for result_file in get_all_results_files_in_folder(dataset,run):
            file_path = os.path.join(DataManager.get_results_folder(),dataset,"fully_train", run, result_file)
            train_config = result_file.split("fm")[0].replace("_", " ")
            train_config = "NONE" if len(train_config) == 0 else train_config

            fm = result_file.split("fm")[1].replace(".txt", "").replace(",", ".")

            with open(file_path) as file:
                lines = file.readlines()
                accuracies = get_accuracies(lines)
                max_acc = max(accuracies)

                if run not in data:
                    data[run] = {}
                if train_config not in data[run]:
                    data[run][train_config] = []
                data[run][train_config].append((float(fm), max_acc))

    return data


def plot_fm_acc_tuples(dataset):
    data = get_fm_acc_tuples(dataset)

    for run in data.keys():
        for config in data[run].keys():
            tuples = data[run][config]
            if len(tuples) < 2:
                continue

            tuples = sorted(tuples, key=lambda x: x[0])
            fms = [fm for (fm, y) in tuples]
            accs = [acc for (x, acc) in tuples]
            name = run + ":" + config if config != "NONE" else run
            plt.plot(fms, accs, label=name)

    handles, labels = plt.gca().get_legend_handles_labels()
    plt.gca().legend(handles, labels)
    plt.xlabel("Feature Multiplier")
    plt.ylabel("Max Accuracy %")
    plt.title("Accuracy Of Fully Trained DNN At differing Feature Multiplication Values")
    plt.show()


if __name__ == "__main__":
    dataset = "CIFAR-10"
    print_max_accuracies(dataset)
    # plot_fm_acc_tuples(dataset)
