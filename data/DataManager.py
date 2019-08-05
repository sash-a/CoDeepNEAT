import inspect, os
from src.Config import Config
import pickle


def create_runs_folder():
    # print("creating runs folder at:", get_run_folder())

    if not os.path.exists(get_Datasets_folder()):
        os.makedirs(get_Datasets_folder())

    if not os.path.exists(get_run_folder()):
        os.makedirs(get_run_folder())
        os.makedirs(get_Graphs_folder())
        os.makedirs(get_Logs_folder())


def get_Logs_folder(run_name=None):
    return os.path.join(get_run_folder(run_name), "Logs")


def get_Graphs_folder(run_name=None):
    return os.path.join(get_run_folder(run_name), "Graphs")


def get_run_folder(run_name=None):
    if run_name is None:
        return os.path.join(get_data_folder(), "runs", Config.run_name)
    else:
        return os.path.join(get_data_folder(), "runs", run_name)


def get_Datasets_folder():
    if Config.data_path == "":
        return os.path.join(get_data_folder(), "Datasets")
    else:
        return Config.data_path


def get_DataEfficiencyResults_folder():
    return os.path.join(get_data_folder(), "DataEfficiencyResults")


def get_data_folder():
    return os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))


def save_generation_state(generation):
    generation_state_file_name = os.path.join(get_run_folder(), "GenerationState.pickle")
    pickle_out = open(generation_state_file_name, "wb")
    pickle.dump(generation, pickle_out)
    pickle_out.close()


def load_generation_state(run_name):
    generation_state_file_name = os.path.join(get_run_folder(run_name), "GenerationState.pickle")
    pickle_in = open(generation_state_file_name, "rb")
    return pickle.load(pickle_in)


create_runs_folder()
