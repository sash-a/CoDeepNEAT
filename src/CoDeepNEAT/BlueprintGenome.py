from src.NEAT.Genome import Genome
from src.Blueprint.Blueprint import BlueprintNode
from src.NEAT.Connection import Connection
from src.CoDeepNEAT.BlueprintNEATNode import BlueprintNEATNode


class BlueprintGenome(Genome):

    def __init__(self, connections, nodes):
        super(BlueprintGenome, self).__init__(connections, nodes)
        # TODO clear after eval
        self.modules_used = []  # holds ref to module individuals used - can multiple represent
        self.fitness = []

    def to_blueprint(self):
        """
        turns blueprintNEATNodes from self.nodes into BlueprintNodes and connects them into a graph with self.connections
        :return: the blueprint graph this individual represents
        """
        return super().to_phenotype(BlueprintNode)

    def _mutate_add_node(self, conn: Connection, mutations: dict, innov: int, node_id: int,
                         MutationType=BlueprintNEATNode):
        return super()._mutate_add_node(conn, mutations, innov, node_id, MutationType)  # innov, node_id

    def report_fitness(self, fitness):
        self.fitness = fitness

    def clear(self):
        self.modules_used = []
