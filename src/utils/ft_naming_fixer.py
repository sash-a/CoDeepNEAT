from __future__ import annotations

import os

from configuration import config
from runs.runs_manager import get_fully_train_folder_path, get_run_folder_path, __get_runs_folder_path
from src.analysis.run import Run, get_run


def fix(run_name):
    run: Run = get_run(run_name)
    best_blueprints = run.get_most_accurate_blueprints(config.fully_train_best_n_blueprints)

    print(f'best blueprints ({len(best_blueprints)}): {[b[0].id for b in best_blueprints]}')
    best = 1
    for bp, _ in best_blueprints:

        for fm in [1, 3, 5]:
            old_file_name = f'bp-{bp.id}_fm-{fm}.model'
            new_file_name = f'bp-{bp.id}_fm-{fm}-best-{best}.model'

            old_file_path = os.path.join(get_fully_train_folder_path(run_name), old_file_name)
            new_file_path = os.path.join(get_fully_train_folder_path(run_name), new_file_name)

            if os.path.exists(old_file_path):
                print(f'renaming: {old_file_name} to {new_file_name}')
                os.rename(old_file_path, new_file_path)

        best += 1


if __name__ == '__main__':
    for run in os.listdir(__get_runs_folder_path()):
        if 'base' in run or 'elite' in run:
            fix(get_run_folder_path(run))

    # fix(get_run_folder_path('elite_1'))
