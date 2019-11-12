from abc import ABC, abstractmethod

from src2.Genotype.NEAT import Genome
from src2.Genotype.NEAT import MutationRecord


class Mutator(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def mutate(self, genome: Genome, mutation_record: MutationRecord):
        pass
