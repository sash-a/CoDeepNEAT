import json
from os.path import join

import wandb

import configuration


class InternalConfig:
    def __init__(self):
        self.state = 'evo'  # evo | ft | finished
        self.running = True
        self.finished = False

        self.ft_started = False
        self.generation = 0

    def save(self, run_name: str, wandb_save=True):
        import runs.runs_manager as runs_manager
        file_path = join(runs_manager.get_run_folder_path(run_name), 'internal_config.json')

        with open(file_path, 'w+') as f:
            json.dump(self.__dict__, f, indent=2)

        if configuration.config.use_wandb and wandb_save:
            try:
                wandb.save(file_path)
            except ValueError:
                print('Error: You must call `wandb.init` before calling save. This happens because wandb is not '
                      'initialized in the main thread in fully training. If you were not fully training this should '
                      'be investigated, otherwise ignore it')

    def load(self, run_name: str):
        import runs.runs_manager as runs_manager
        file_path = join(runs_manager.get_run_folder_path(run_name), 'internal_config.json')

        with open(file_path, 'r+') as f:
            for k, v in json.load(f).items():
                if k in self.__dict__:
                    self.__dict__[k] = v

    def on_exit(self):
        print('Exiting!')
        self.running = False
        self.save(configuration.config.run_name)
        print('Exited!')
