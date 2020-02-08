import json
from os.path import join

import wandb

import runs.runs_manager as runs_manager
import configuration


class InternalConfig:
    def __init__(self):
        self.state = 'evo'  # evo | ft
        self.running = True

        self.ft_epoch = 0
        self.generation = 0

    def save(self, run_name: str):
        file_path = join(runs_manager.get_run_folder_path(run_name), 'internal_config.json')

        with open(file_path, 'w+') as f:
            json.dump(self.__dict__, f, indent=2)

        if configuration.config.use_wandb:
            wandb.save(file_path)

    def load(self, run_name: str):
        file_path = join(runs_manager.get_run_folder_path(run_name), 'internal_config.json')

        with open(file_path, 'w+') as f:
            for k, v in json.load(f).items():
                if k in self.__dict__:
                    self.__dict__[k] = v

    def on_exit(self):
        self.running = False
        if self.generation - 1 == configuration.config.n_generations:
            self.state = 'ft'

        self.save(configuration.config.run_name)
