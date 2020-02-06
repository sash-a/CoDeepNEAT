import inspect
import os

def get_datasets_folder():
    return os.path.join(get_data_folder(), "Datasets")


def get_data_folder():
    return os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

