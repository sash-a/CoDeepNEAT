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

    def mutate(self):
        raise NotImplementedError("method must be implemented in sub class")

    def set_value(self, value):
        raise NotImplementedError("method must be implemented in sub class")
