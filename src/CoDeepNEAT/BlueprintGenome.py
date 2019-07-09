from src.NEAT.Genome import Genome
from src.Blueprint.Blueprint import BlueprintNode
from src.NEAT.Connection import Connection
from src.CoDeepNEAT.BlueprintNEATNode import BlueprintNEATNode


class BlueprintGenome(Genome):

    def __init__(self, connections, nodes, objectives=2):
        super(BlueprintGenome, self).__init__(connections, nodes)
        self.modules_used = []  # holds ref to module individuals used - can multiple represent
        self.fitness = [0 for _ in range(objectives)]

    def to_blueprint(self):
        """
        turns blueprintNEATNodes from self.nodes into BlueprintNodes and connects them into a graph with self.connections
        :return: the blueprint graph this individual represents
        """
        return super().to_phenotype(BlueprintNode)

    def _mutate_add_node(self, conn: Connection, mutations: dict, innov: int, node_id: int,
                         MutationType=BlueprintNEATNode):
        return super()._mutate_add_node(conn, mutations, innov, node_id, MutationType)  # innov, node_id

    def report_fitness(self, *fitnesses):
        # print('recieved fitness:', fitnesses)
        for i, fitness in enumerate(fitnesses):
            self.fitness[i] = fitness

    def clear(self ):
        self.modules_used = []

    def reset_number_of_module_species(self,num_module_species):
        for node in self.nodes:
            node.set_species_upper_bound(num_module_species)