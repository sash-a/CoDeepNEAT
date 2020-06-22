from src.analysis.reporters.base_reporter import BaseReporter
from src.genotype.cdn.genomes.blueprint_genome import BlueprintGenome
from src.phenotype.neural_network.neural_network import Network


class ReporterSet:
    def __init__(self, *reporters: BaseReporter):
        self.reporters = list(reporters)

    def on_start_train(self, blueprint: BlueprintGenome):
        for reporter in self.reporters:
            reporter.on_start_train(blueprint)

    def on_end_train(self, blueprint: BlueprintGenome, acc: float):
        for reporter in self.reporters:
            reporter.on_end_train(blueprint, acc)

    def on_start_epoch(self, model: Network, epoch: int):
        for reporter in self.reporters:
            reporter.on_start_epoch(model, epoch)

    def on_end_epoch(self, model: Network, epoch: int, loss: float, acc: float):
        for reporter in self.reporters:
            reporter.on_end_epoch(model, epoch, loss, acc)

    def on_start_batch(self, batch_idx: int, loss: float):
        for reporter in self.reporters:
            reporter.on_start_batch(batch_idx, loss)

    def on_end_batch(self, batch_idx: int, loss: float):
        for reporter in self.reporters:
            reporter.on_end_batch(batch_idx, loss)
