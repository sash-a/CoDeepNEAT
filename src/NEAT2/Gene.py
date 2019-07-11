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
    pass


class ConnectionGene(Gene):
    pass
