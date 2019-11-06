from src2.Genotype.Mutagen.Mutagen import Mutagen


class Variable(Mutagen):

    def __init__(self, name, current_value, start_range, end_range, mutation_chance):
        super().__init__(name, mutation_chance)

        self.start_range = start_range
        self.end_range = end_range
        self.current_value = min(end_range, max(start_range, current_value))

    def __str__(self):
        return self.name + ": " + repr(self.current_value)

    def get_current_value(self):
        return self.current_value

    def set_value(self, value):
        self.current_value = value
