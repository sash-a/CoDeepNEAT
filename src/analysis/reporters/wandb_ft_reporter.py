import wandb

from runs.runs_manager import save_config
from src.analysis.reporters.base_reporter import BaseReporter
from src.genotype.cdn.genomes.blueprint_genome import BlueprintGenome
from src.phenotype.neural_network.evaluator.eval_utils import RETRY
from src.phenotype.neural_network.evaluator.training_results import TrainingResults
from src.phenotype.neural_network.neural_network import Network
from configuration import config, internal_config
from src.utils.wandb_utils import resume_ft_run, new_ft_run


class WandbFTReporter(BaseReporter):
    """Reporter for logging to wandb during fully train"""

    def __init__(self, fm: float, best: int):
        """
        @param fm: feature multiplier: how much bigger or smaller to make each layer
        @param best: the ranking of the network in evolution - ie best = 1 mean that network got the highest accuracy
         in evolution
        """
        self.fm = fm
        self.best = best
        self.training_results = TrainingResults()

    def on_start_train(self, blueprint: BlueprintGenome):
        pass

    def on_end_train(self, blueprint: BlueprintGenome, accuracy: float):
        """Creates the wandb run and logs all relevant data if run was not a 'dud'"""

        fm_tag = f'FM={self.fm}'  # wandb tag for feature mul so that we can tell the difference
        best_tag = f'BEST={self.best}'  # wandb tag for Nth best network in evolution

        if RETRY in config.wandb_tags:
            config.wandb_tags.remove(RETRY)

        if config.use_wandb:
            config.wandb_tags = list(set([tag for tag in config.wandb_tags if 'FM=' not in tag and 'BEST=' not in tag]))
            config.wandb_tags += [fm_tag, best_tag]
            if accuracy == RETRY:  # If it was a retry then we know it was a bad run, so leave it out as it is an outlier
                config.wandb_tags += [RETRY]

            new_ft_run(True)

            # specific options for wandb grouping
            wandb.config['fm'] = self.fm
            wandb.config['best'] = self.best

        for epoch, loss in enumerate(self.training_results.losses):
            log = {'loss': loss}
            if epoch in self.training_results.accuracy_epochs:
                acc = self.training_results.accuracies[self.training_results.accuracy_epochs.index(epoch)]
                log.update({f'accuracy_fm_{self.fm}': acc})

            wandb.log(log)

        wandb.join()
        if accuracy == RETRY or RETRY in config.wandb_tags:
            config.wandb_tags.remove(RETRY)
        config.wandb_tags.remove(fm_tag)
        config.wandb_tags.remove(best_tag)

    def on_start_epoch(self, model: Network, epoch: int):
        pass

    def on_end_epoch(self, model: Network, epoch: int, loss: float, acc: float):
        if acc != -1:
            self.training_results.add_accuracy(acc, epoch)

        self.training_results.add_loss(loss)

        internal_config.ft_epoch = epoch
        save_config(config.run_name, use_wandb_override=False)

    def on_start_batch(self, batch_idx: int, loss: float):
        pass

    def on_end_batch(self, batch_idx: int, loss: float):
        pass
