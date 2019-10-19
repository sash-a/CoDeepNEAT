from src2.Genotype.Mutagen.Mutagen import Mutagen


class Variable(Mutagen):

    def __init__(self, name,current_value, start_range, end_range, mutation_chance):
        super(name, mutation_chance)

        self.current_value = current_value
        self.start_range = start_range
        self.end_range = end_range

