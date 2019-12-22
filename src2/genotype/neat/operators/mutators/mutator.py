from abc import ABC, abstractmethod

from src2.genotype.neat import genome
from src2.genotype.neat import mutation_record


class Mutator(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def mutate(self, genome: genome, mutation_record: mutation_record):
        pass
