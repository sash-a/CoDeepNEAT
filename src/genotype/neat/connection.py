# TESTING
from src.genotype.mutagen.option import Option
from src.genotype.neat.gene import Gene


class Connection(Gene):
    def __init__(self, id: int, from_node: int, to_node: int, enabled=True):
        super().__init__(id)

        self.from_node_id: int = from_node
        self.to_node_id: int = to_node

        self.enabled: Option = Option("connection_enabled", True, False, current_value=enabled)

    def __repr__(self):
        return 'Conn: ' + repr(self.from_node_id) + "->" + repr(self.to_node_id) + ' (' + repr(self.enabled()) + ')'

    def get_all_mutagens(self):
        return [self.enabled]
