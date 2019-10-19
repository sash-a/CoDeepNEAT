import random

import math

from src2.Genotype.Mutagen.Variable import Variable


class ContinuousVariable(Variable):

    def __init__(self, name, current_value: float, start_range: float, end_range: float, mutation_chance):
        super(name, current_value, start_range, end_range, mutation_chance)

    def mutate(self):
        if random.random() > self.mutation_chance:
            return

        range = self.end_range - self.start_range

        if random.random() < 0.25:
            # random reset
            self.current_value = random.uniform(self.start_range, self.end_range)
            return
        else:
            # random deviation
            deviation_magnitude = math.pow(random.random(), 4)  # TODO find best value
            deviation_dir = (1 if random.choice(True, False) else -1)

            new_current_value = self.current_value + deviation_dir * deviation_magnitude * range
            self.current_value = self.start_range + ((new_current_value - self.start_range) % range)
