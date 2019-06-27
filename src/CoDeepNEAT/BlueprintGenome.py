from src.NEAT.Genotype import Genome
from src.Blueprint.Blueprint import BlueprintNode


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

        blueprint_graph_node_map = {}
        root_node = None
        """initialises blueprint nodes and maps them to their genes"""
        for blueprint_neat_node in self.nodes:
            blueprint_graph_node_map[blueprint_neat_node] = BlueprintNode(blueprint_neat_node, self)
            if (blueprint_neat_node.is_input_node):
                root_node = blueprint_graph_node_map[blueprint_neat_node]
        """connects the blueprint nodes as indicated by the genome"""
        for connection in self.connections:
            parent = blueprint_graph_node_map[connection.from_node]
            child = blueprint_graph_node_map[connection.to_node]

            parent.add_child(child)

        return root_node

    def report_fitness(self, fitness):
        self.fitness = fitness

    def clear(self):
        self.modules_used = []
