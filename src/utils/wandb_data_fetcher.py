from typing import List, Set
import wandb

from runs.runs_manager import get_generations_folder_path, get_run_folder_path, get_fully_train_folder_path, \
    set_up_run_folder, load_config


def fetch_run(run_id: str = '', run_path: str = '') -> wandb.wandb_run:
    if not run_path:
        run_path = 'codeepneat/cdn/' + run_id
    return wandb.Api().run(run_path)


def fetch_run_name(run_id: str = '', run_path: str = ''):
    return fetch_run(run_id, run_path).config['run_name']


def download_run(run_id: str = '', run_path: str = '', replace: bool = False) -> str:
    run_name = fetch_run_name(run_id, run_path)
    set_up_run_folder(run_name)

    for file in fetch_run(run_id, run_path).files():
        if file.name.endswith('.png'):
            continue

        if 'generation' in file.name:
            file.download(replace=replace, root=get_generations_folder_path(run_name))
        elif file.name.endswith('.model'):
            file.download(replace=replace, root=get_fully_train_folder_path(run_name))
        else:
            file.download(replace=replace, root=get_run_folder_path(run_name))

    return run_name


def download_generations(run_id: str = '', run_path: str = '', replace: bool = False, run_name=''):
    if run_name == '':
        run_name = fetch_run_name(run_id, run_path)

    for file in fetch_run(run_id, run_path).files():
        if 'generation' in file.name:
            file.download(replace=replace, root=get_generations_folder_path(run_name))


def download_config(run_id: str = '', run_path: str = '', replace: bool = False, run_name=''):
    if run_name == '':
        run_name = fetch_run_name(run_id, run_path)

    for file in fetch_run(run_id, run_path).files():
        if file.name == 'config.json':
            file.download(replace=replace, root=get_run_folder_path(run_name))
            load_config(run_name)


def download_model(run_id: str = '', run_path: str = '', replace: bool = False, run_name=''):
    if run_name == '':
        run_name = fetch_run_name(run_id, run_path)

    for file in fetch_run(run_id, run_path).files():
        if file.name.endswith('.model'):
            file.download(replace=replace, root=get_fully_train_folder_path(run_name))


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


def fix_untagged_runs(extra_tags: List[str] = []):
    """
    Adds wandb tags to all runs that do not have them (these tags are taken from the config file associated with the
     runs name)
    :param extra_tags: tags to be added on to all untagged runs
    """
    from configuration import config

    for run in wandb.Api().runs(path='codeepneat/cdn', order="-created_at"):
        tags = run.config['wandb_tags']
        if tags:
            continue

        start = run.name.index('_')
        end = run.name.rindex('_')
        name = run.name[start + 1:end] if len(run.name) > 6 else run.name[:end]

        new_tags = list(set(config.read_option(name, 'wandb_tags') + extra_tags))
        print(f'Adding tags: {new_tags} to run: {name}')

        run.config['wandb_tags'] = new_tags
        run.update()



if __name__ == '__main__':
    # fetch_config(run_path='codeepneat/cdn/test2019-12-25_599245')
    # print(fetch_run_name(run_id="base2020-01-08_253129"))
    # print(fetch_run('homerun2020-01-11_315871').history())
    # download_run(run_id="base2020-01-08_253129")
    # fix_untagged_runs()
    import os

    for root, dirs, files in os.walk('/home/sasha/wandb/'):
        for name in dirs:
            if 'lr' in name:
                print(os.path.join(root, name))
                start = name.index('base_base')
                path = name[start:]

                # wandb.restore(os.path.join(root, name))
                # wandb.init()
                # wandb.log({})
                # break
