from src.NEAT.Mutagen import Mutagen
from enum import Enum


class Gene:

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


class NodeType(Enum):
    INPUT = 0
    HIDDEN = 1
    OUTPUT = 2


class NodeGene(Gene):
    def __init__(self, id, type: NodeType = NodeType.HIDDEN):
        super().__init__(id)

        self.height = -1
        self.node_type = type

    def is_output_node(self):
        return self.node_type == NodeType.OUTPUT

    def is_input_node(self):
        return self.node_type == NodeType.INPUT


class ConnectionGene(Gene):
    def __init__(self, id, from_node: int, to_node: int):
        super().__init__(id)

        self.from_node: int = from_node
        self.to_node: int = to_node

        self.enabled = Mutagen(True, False, discreet_value=True)

    def __repr__(self):
        return "Conn(" + repr(self.from_node) + "->" + repr(self.to_node) + ")"

    def mutate_add_node(self, mutation_record, genome):
        mutation = self.id
        # print("adding node to connection",self)
        if mutation_record.exists(mutation):
            mutated_node_id = mutation_record.mutations[mutation]
            mutated_from_conn_id = mutation_record.mutations[(self.from_node, mutated_node_id)]
            mutated_to_conn_id = mutation_record.mutations[(mutated_node_id, self.to_node)]
            # print("mutated node already existed - id's: node:",mutated_node_id,"connections:",mutated_from_conn_id,mutated_to_conn_id)
            if mutated_node_id in genome._nodes:  # this connection has already created a new node
                return  #TODO retry

        else:
            mutated_node_id = mutation_record.add_mutation(mutation)
            mutated_from_conn_id = mutation_record.add_mutation((self.from_node, mutated_node_id))
            mutated_to_conn_id = mutation_record.add_mutation((mutated_node_id, self.to_node))
            # print("mutated node novel - id's: node:",mutated_node_id,"connections:",mutated_from_conn_id,mutated_to_conn_id)

        NodeType = type(list(genome._nodes.values())[0])
        mutated_node = NodeType(mutated_node_id)
        mutated_node.height = (genome._nodes[self.from_node].height + genome._nodes[self.to_node].height)/2
        genome.add_node(mutated_node)

        mutated_from_conn = ConnectionGene(mutated_from_conn_id, self.from_node, mutated_node_id)
        mutated_to_conn = ConnectionGene(mutated_to_conn_id, mutated_node_id, self.to_node)

        genome.add_connection(mutated_from_conn)
        genome.add_connection(mutated_to_conn)

        self.enabled.set_value(False)
