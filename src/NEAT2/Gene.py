from src.NEAT.Mutagen import Mutagen

class Gene:
    _next_id = 0

    def __init__(self, id=None):
        if id is None:
            self.id = Gene._get_new_id()
        else:
            self.id = id

    @classmethod
    def _get_new_id(cls):
        cls._next_id += 1
        return cls._next_id

    def __cmp__(self, other):
        if self.id < other.id:
            return -1
        elif self.id == other.id:
            return 0
        elif self.id > other.id:
            return 1


class NodeGene(Gene):
    def __init__(self):
        super().__init__()

        self.height = -1
        self.node_type = ""

    def is_output_node(self):
        pass

    def is_input_node(self):
        pass


class ConnectionGene(Gene):
    def __init__(self, from_node: NodeGene, to_node: NodeGene):
        super().__init__()

        self.from_node: NodeGene = from_node
        self.to_node: NodeGene = to_node

        self.enabled = Mutagen(True, False, discreet_value=True)
