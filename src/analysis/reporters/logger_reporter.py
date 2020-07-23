import logging
import sys
from os.path import join

from configuration import config
from runs.runs_manager import get_run_folder_path
from src.analysis.reporters.base_reporter import BaseReporter
from src.genotype.cdn.genomes.blueprint_genome import BlueprintGenome
from src.phenotype.neural_network.evaluator.eval_utils import RETRY
from src.phenotype.neural_network.neural_network import Network


class LoggerReporter(BaseReporter):
    def __init__(self, fm: float, best: int, attempt: int):
        """
        @param fm: feature multiplier: how much bigger or smaller to make each layer
        @param best: the ranking of the network in evolution - ie best = 1 mean that network got the highest accuracy
         in evolution
        """
        file = join(get_run_folder_path(config.run_name), f'fm={fm}_best={best}_attempt={attempt}.log')
        self.logger = logging.getLogger(file)
        self.logger.addHandler(logging.FileHandler(file, 'a'))
        self.logger.setLevel(logging.DEBUG)

        self.logger.info(f'fm:{fm}')
        self.logger.info(f'best:{best}')
        self.logger.info(f'attempt:{attempt}')
        self.logger.info(f'config:{config.__dict__}')

        print(f'INITIALIZED FT LOGGER. DIR={file}')
        sys.stdout.flush()

    def on_start_train(self, blueprint: BlueprintGenome):
        self.logger.info(f'blueprint:{blueprint.id}')

    def on_end_train(self, blueprint: BlueprintGenome, accuracy: float):
        self.logger.info(f'accP{RETRY}')

    def on_start_epoch(self, model: Network, epoch: int):
        pass

    def on_end_epoch(self, model: Network, epoch: int, loss: float, acc: float):
        self.logger.info(f'epoch:{epoch}')
        if acc != -1:
            self.logger.info(f'acc:{acc}')

        self.logger.info(f'loss:{loss}')

    def on_start_batch(self, batch_idx: int, loss: float):
        pass

    def on_end_batch(self, batch_idx: int, loss: float):
        pass
