from abc import ABC, abstractmethod


class Gene(ABC):
    """base class for a neat node and connection"""

    def __init__(self, id: int):
        self.id: int = id

    def __eq__(self, other):
        return self.id == other.id

    def __hash__(self):
        return self.id

    @abstractmethod
    def get_all_mutagens(self):
        raise NotImplementedError('get_all_mutagens should not be called from Gene')

    def mutate(self):
        for mutagen in self.get_all_mutagens():
            mutagen.mutate()
