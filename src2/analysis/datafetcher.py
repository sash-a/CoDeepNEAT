from typing import List, Dict, Set
import wandb

from runs.runs_manager import get_generations_folder_path, get_run_folder_path

from src2.configuration import config

# TODO move to wandb utils
def fetch_run(run_id: str = '', run_path: str = '') -> wandb.run:
    if not run_path:
        run_path = 'codeepneat/cdn/' + run_id
    return wandb.Api().run(run_path)


def fetch_generations(run_id: str = '', run_path: str = '', replace: bool = False):
    for file in fetch_run(run_id, run_path).files():
        if 'generation' in file.name:
            file.download(replace=replace, root=get_generations_folder_path(config.run_name))


def fetch_config(run_id: str = '', run_path: str = '', replace: bool = False):
    for file in fetch_run(run_id, run_path).files():
        if file.name == 'config.json':
            print(get_run_folder_path(config.run_name))
            file.download(replace=replace, root=get_run_folder_path(config.run_name))


def get_all_metrics(run_id: str = '', run_path: str = '') -> Set[str]:
    metric_names: Set[str] = set()
    for gen in fetch_run(run_id, run_path).history():
        metric_names.update(gen.keys())

    return metric_names


def get_metric(metric_name: str, run_id: str = '', run_path: str = '') -> List:
    gathered_metric: List = []
    for gen in fetch_run(run_id, run_path).history():
        gathered_metric.append(gen[metric_name])

    return gathered_metric


if __name__ == '__main__':
    fetch_config(run_path='codeepneat/cdn/test2019-12-25_599245')
