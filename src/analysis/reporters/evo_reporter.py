from src.analysis.reporters.base_reporter import BaseReporter
from src.genotype.cdn.genomes.blueprint_genome import BlueprintGenome
from src.phenotype.neural_network.neural_network import Network


class EvoReporter(BaseReporter):

    def on_start_train(self, blueprint: BlueprintGenome):
        pass

    def on_end_train(self, blueprint: BlueprintGenome, accuracy: float):
        pass

    def on_start_epoch(self, model: Network, epoch: int):
        pass

    def on_end_epoch(self, model: Network, epoch: int, loss: float, acc: float):
        pass

    def on_start_batch(self, batch_idx: int, loss: float):
        pass

    def on_end_batch(self, batch_idx: int, loss: float):
        pass
