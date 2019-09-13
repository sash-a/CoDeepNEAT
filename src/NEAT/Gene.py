import copy
from enum import Enum

from src.NEAT.Mutagen import Mutagen


class Gene:
    """a gene is the general form of a neat node or connection"""

    def __init__(self, id):
        self.id = id

    def __cmp__(self, other):
        if self.id < other.id:
            return -1
        elif self.id == other.id:
            return 0
        elif self.id > other.id:
            return 1

    def __eq__(self, other):
        return self.id == other.id

    def __hash__(self):
        return self.id

    def get_all_mutagens(self):
        raise NotImplementedError("Implement get all mutagens in super classes")

    def mutate(self, magnitude=1):
        mutated = False
        for mutagen in self.get_all_mutagens():
            mutated = mutagen.mutate(magnitude=magnitude) or mutated

        return mutated

    def breed(self, other):
        """returns chlid of self and other by cloning self and inheriting mutagen values from other to the clone"""
        clone = copy.deepcopy(self)
        for (best, worst) in zip(clone.get_all_mutagens(), other.get_all_mutagens()):
            best.inherit(worst)

        return clone


class NodeType(Enum):
    INPUT = 0
    HIDDEN = 1
    OUTPUT = 2


class NodeGene(Gene):
    """the general form of blueprintNeatNode ect"""

    def __init__(self, id, type: NodeType = NodeType.HIDDEN):
        super().__init__(id)

        self.height = -1
        self.node_type = type

    def is_output_node(self):
        return NodeType.OUTPUT.value == self.node_type.value

    def is_input_node(self):
        return NodeType.INPUT.value == self.node_type.value

    def get_all_mutagens(self):
        return []


class ConnectionGene(Gene):
    """Neat connection between nodes"""

    def __init__(self, id, from_node: int, to_node: int):
        super().__init__(id)

        self.from_node: int = from_node
        self.to_node: int = to_node

        self.enabled = Mutagen(True, False, discreet_value=True)

    def __repr__(self):
        return 'Conn: ' + repr(self.from_node) + "->" + repr(self.to_node) + ' (' + repr(self.enabled()) + ')'

    def get_all_mutagens(self):
        return [self.enabled]

    def mutate_add_node(self, mutation_record, genome):
        """Adds a node on a connection and updates the relevant genome"""
        mutation = self.id
        if mutation_record.exists(mutation):
            mutated_node_id = mutation_record.mutations[mutation]
            mutated_from_conn_id = mutation_record.mutations[(self.from_node, mutated_node_id)]
            mutated_to_conn_id = mutation_record.mutations[(mutated_node_id, self.to_node)]
            if mutated_node_id in genome._nodes:  # this connection has already created a new node
                return  # TODO retry (not important)

        else:
            mutated_node_id = mutation_record.add_mutation(mutation)
            mutated_from_conn_id = mutation_record.add_mutation((self.from_node, mutated_node_id))
            mutated_to_conn_id = mutation_record.add_mutation((mutated_node_id, self.to_node))

        NodeType = type(list(genome._nodes.values())[0])
        mutated_node = NodeType(mutated_node_id)
        mutated_node.height = (genome._nodes[self.from_node].height + genome._nodes[self.to_node].height) / 2
        genome.add_node(mutated_node)

        mutated_from_conn = ConnectionGene(mutated_from_conn_id, self.from_node, mutated_node_id)
        mutated_to_conn = ConnectionGene(mutated_to_conn_id, mutated_node_id, self.to_node)

        genome.add_connection(mutated_from_conn)
        genome.add_connection(mutated_to_conn)

        self.enabled.set_value(False)
        return mutated_node
