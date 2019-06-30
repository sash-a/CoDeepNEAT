from src.NEAT.Genotype import Genome
from src.Module.ModuleNode import ModuleNode
from src.NEAT.Connection import Connection
from src.NEAT.Mutation import NodeMutation
from src.CoDeepNEAT.ModuleNEATNode import ModulenNEATNode
import copy


class ModuleGenome(Genome):

    def __init__(self, connections, nodes):
        super(ModuleGenome, self).__init__(connections, nodes)
        self.fitness_reports = 0  # todo zero out
        self.module_node = None  # the module node created from this gene

    def to_module_node(self):
        """
        returns the stored module_node of this gene, or generates and returns it if module_node is null
        :return: the module graph this individual represents
        """
        if self.module_node is not None:
            print("module genome already has module - returning a copy")
            return copy.deepcopy(self.module_node)

        # needs to generate the module_node

        module_graph_node_map = {}
        root_node = None
        # initialises blueprint nodes and maps them to their genes
        for module_neat_node in self.nodes:
            module_graph_node_map[module_neat_node.id] = ModuleNode(module_neat_node, self)
            if module_neat_node.is_input_node():
                root_node = module_graph_node_map[module_neat_node.id]

        # connects the blueprint nodes as indicated by the genome
        for connection in self.connections:
            if not connection.enabled:
                continue

            parent = module_graph_node_map[connection.from_node.id]
            child = module_graph_node_map[connection.to_node.id]

            parent.add_child(child)

        self.module_node = root_node
        return copy.deepcopy(root_node)

    def _mutate_add_node(self, conn: Connection, curr_gen_mutations: set, innov: int, node_id: int):
        conn.enabled = False

        mutated_node = ModulenNEATNode(node_id + 1, conn.from_node.midpoint(conn.to_node))
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

        print('mutated node', node_id, mutated_from_conn, mutated_to_conn)

        return innov, node_id

    def report_fitness(self, fitness):
        self.fitness = (self.fitness * self.fitness_reports + fitness) / (self.fitness_reports + 1)
        self.fitness_reports += 1

    def clear(self):
        self.fitness_reports = 0
        self.fitness = 0
        self.module_node = None
