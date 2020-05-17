from typing import List

import wandb


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


def fix_trimmed_name(path):
    """Adds trimmed name to wandb runs that have null one"""
    from configuration import config

    for run in wandb.Api().runs(path=path, order="-created_at"):
        if 'trimmed_name' not in run.config:
            run.config['trimmed_name'] = run.name[:-2] if run.name[-1].isdigit() else run.name
        if 'run_name' not in run.config:
            run.config['run_name'] = run.name

        run.update()


if __name__ == '__main__':
    fix_trimmed_name('cdn/codeepneat')
    fix_trimmed_name('cdn/cdn_fully_train')
