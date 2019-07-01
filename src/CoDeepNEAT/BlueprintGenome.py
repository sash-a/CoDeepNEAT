from src.NEAT.Genotype import Genome
from src.Blueprint.Blueprint import BlueprintNode
from src.NEAT.Connection import Connection
from src.NEAT.Mutation import NodeMutation
from src.CoDeepNEAT.BlueprintNEATNode import BlueprintNEATNode


class BlueprintGenome(Genome):

    def __init__(self, connections, nodes):
        super(BlueprintGenome, self).__init__(connections, nodes)
        # TODO clear after eval
        self.modules_used = []  # holds ref to module individuals used - can multiple represent

    def to_blueprint(self):
        """
        turns blueprintNEATNodes from self.nodes into BlueprintNodes and connects them into a graph with self.connections
        :return: the blueprint graph this individual represents
        """
        return super().to_phenotype(BlueprintNode)

    def _mutate_add_node(self, conn: Connection, curr_gen_mutations: set, innov: int, node_id: int):
        conn.enabled = False

        mutated_node = BlueprintNEATNode(node_id + 1, conn.from_node.midpoint(conn.to_node))
        mutated_from_conn = Connection(conn.from_node, mutated_node)
        mutated_to_conn = Connection(mutated_node, conn.to_node)

        mutation = NodeMutation(mutated_node.id, mutated_from_conn, mutated_to_conn)

        innov, node_id = super()._check_node_mutation(mutation,
                                                      mutated_node,
                                                      mutated_from_conn,
                                                      mutated_to_conn,
                                                      curr_gen_mutations, innov,
                                                      node_id)

        self.add_connection(mutated_from_conn)
        self.add_connection(mutated_to_conn)
        self.add_node(mutated_node)

        print('mutated blueprint node', node_id, mutated_from_conn, mutated_to_conn)

        return innov, node_id

    def report_fitness(self, fitness):
        self.fitness = fitness

    def clear(self):
        self.modules_used = []
