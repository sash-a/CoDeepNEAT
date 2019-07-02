from src.NEAT.Genome import Genome
from src.Module.ModuleNode import ModuleNode
from src.NEAT.Connection import Connection
from src.CoDeepNEAT.ModuleNEATNode import ModulenNEATNode
import copy


class ModuleGenome(Genome):

    def __init__(self, connections, nodes):
        super(ModuleGenome, self).__init__(connections, nodes)
        self.fitness_reports = 0
        self.module_node = None  # the module node created from this gene

    def to_module(self):
        """
        returns the stored module_node of this gene, or generates and returns it if module_node is null
        :return: the module graph this individual represents
        """
        if self.module_node is not None:
            print("module genome already has module - returning a copy")
            return copy.deepcopy(self.module_node)

        return copy.deepcopy(super().to_phenotype(ModuleNode))

    def _mutate_add_node(self, conn: Connection, mutations: dict, innov: int, node_id: int,
                         MutationType=ModulenNEATNode):
        return super()._mutate_add_node(conn, mutations, innov, node_id, MutationType)  # innov, node_id

    def report_fitness(self, fitness):
        self.fitness = (self.fitness * self.fitness_reports + fitness) / (self.fitness_reports + 1)
        self.fitness_reports += 1

    def clear(self):
        self.fitness_reports = 0
        self.fitness = 0
        self.module_node = None
