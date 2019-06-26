from src.NEAT.Genotype import Genome

class BlueprintGenome(Genome):

    def __init__(self, connections, nodes):
        super(BlueprintGenome, self).__init__(connections,nodes)
        #TODO clear after eval
        self.modules_used = []#holds ref to module individuals used - can multiple represent

    def to_blueprint(self):
        """

        :return: the blueprint graph this individual represents
        """
        pass

    def report_fitness(self, fitness):
        self.fitness = fitness

    def clear(self):
        self.modules_used = []