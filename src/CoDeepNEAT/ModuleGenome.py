from src.NEAT.Genotype import Genome
from src.Module.ModuleNode import ModuleNode
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
            module_graph_node_map[module_neat_node] = ModuleNode(module_neat_node, self)
            if module_neat_node.is_input_node():
                root_node = module_graph_node_map[module_neat_node]

        # connects the blueprint nodes as indicated by the genome
        for connection in self.connections:
            parent = module_graph_node_map[connection.from_node]
            child = module_graph_node_map[connection.to_node]

            parent.add_child(child)

        self.module_node = root_node
        return root_node

    def report_fitness(self, fitness):
        self.fitness = (self.fitness * self.fitness_reports + fitness) / (self.fitness_reports + 1)
        self.fitness_reports += 1

    def clear(self):
        self.fitness_reports = 0
        self.fitness = 0
        self.module_node = None
