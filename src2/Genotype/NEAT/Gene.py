
class Gene:
    """a gene is the general form of a neat node or connection"""

    def __init__(self, id: int):
        self.id: int = id

    def __eq__(self, other):
        return self.id == other.id

    def __hash__(self):
        return self.id

    def get_all_mutagens(self):
        raise NotImplementedError("Implement get all mutagens in super classes")

    def mutate(self, magnitude=1):
        for mutagen in self.get_all_mutagens():
            mutagen.mutate(magnitude=magnitude)

