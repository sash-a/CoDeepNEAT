from src2.Genotype.Mutagen.Option import Option
from src2.Genotype.NEAT.Gene import Gene

# TESTING
from src.NEAT.Gene import ConnectionGene


class Connection(Gene):
    def __init__(self, id: int, from_node: int, to_node: int):
        super().__init__(id)

        self.from_node_id: int = from_node
        self.to_node_id: int = to_node

        self.enabled: Option = Option("connection_enabled", True, False, current_value=True)

    def __repr__(self):
        return 'Conn: ' + repr(self.from_node_id) + "->" + repr(self.to_node_id) + ' (' + repr(self.enabled()) + ')'

    def get_all_mutagens(self):
        return [self.enabled]

    def old(self):
        return ConnectionGene(self.id, self.from_node_id, self.to_node_id)
