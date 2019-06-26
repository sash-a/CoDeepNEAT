from src.NEAT.Genotype import Genome


class ModuleGenome(Genome):

    def __init__(self, connections, nodes):
        super(ModuleGenome, self).__init__(connections, nodes)
        self.fitness_reports = 0#todo zero out

    def to_module(self):
        """

        :return: the module graph this individual represents
        """
        pass

    def report_fitness(self, fitness):
        self.fitness = (self.fitness * self.fitness_reports + fitness) / (self.fitness_reports + 1)
        self.fitness_reports += 1

    def clear(self):
        self.fitness_reports = 0