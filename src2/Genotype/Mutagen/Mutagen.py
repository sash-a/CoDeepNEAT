from src2.Genotype.NEAT.Operators.Mutators.MutationReport import MutationReport


class Mutagen:
    """This is the base class for any mutate-able property"""

    def __init__(self, name: str, mutation_chance: float):
        self.name = name
        self.mutation_chance = mutation_chance

    value = property(lambda self: self.get_current_value())

    def get_current_value(self):
        raise NotImplementedError("method must be implemented in sub class")

    def __call__(self):
        return self.get_current_value()

    def mutate(self) -> MutationReport:
        raise NotImplementedError("method must be implemented in sub class")

    def set_value(self, value):
        raise NotImplementedError("method must be implemented in sub class")

    def _interpolate(self, other):
        raise NotImplementedError("method must be implemented in sub class")


def interpolate(mutagen_a: Mutagen, mutagen_b: Mutagen) -> Mutagen:
    if mutagen_a.name != mutagen_b.name:
        raise Exception("cannot interpolate different types of mutagens: " + mutagen_a.name + " and " + mutagen_b.name)

    return mutagen_a._interpolate(mutagen_b)
