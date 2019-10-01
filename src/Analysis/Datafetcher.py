import wandb
from typing import List, Dict, Set


def fetch_history(run_id: str) -> List[Dict]:
    return wandb.Api().run('sash-a/cdn_test/' + run_id).history()


def get_all_metrics(run_id: str) -> Set[str]:
    metric_names: Set[str] = set()
    for gen in fetch_history(run_id):
        metric_names.update(gen.keys())

    return metric_names


def get_metric(run_id: str, metric_name: str) -> List:
    gathered_metric: List = []
    for gen in fetch_history(run_id):
        gathered_metric.append(gen[metric_name])

    return gathered_metric


if __name__ == '__main__':
    print(get_all_metrics('norcbhzn'))
    print(get_metric('norcbhzn', '_step'))
    print(get_metric('norcbhzn', 'accuracies'))
