import random

import math

from src2.Genotype.Mutagen.Variable import Variable


class IntegerVariable(Variable):

    def __init__(self, name, current_value: int, start_range: int, end_range: int, mutation_chance):
        super(name, current_value, start_range, end_range, mutation_chance)

    def mutate(self):
        if random.random() > self.mutation_chance:
            return

        range = self.end_range - self.start_range

        if random.random() < 0.25:
            # random reset
            new_current_value = random.randint(self.start_range, self.end_range)
        else:
            # random deviation
            deviation_magnitude = math.pow(random.random(), 4)
            deviation_direction = (1 if random.choice(True, False) else -1)

            new_current_value = self.current_value + int(deviation_direction * deviation_magnitude * range)

        # making sure value changes
        if new_current_value == self.current_value:
            new_current_value = self.current_value + random.choice(0, 1)

        self.current_value = self.start_range + ((new_current_value - self.start_range) % range)
