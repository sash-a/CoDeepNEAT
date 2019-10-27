import random

import math

from src2.Genotype.Mutagen.Variable import Variable
from src2.Genotype.NEAT.Operators.Mutations.MutationReport import MutationReport


class IntegerVariable(Variable):

    def __init__(self, name, current_value: int, start_range: int, end_range: int, mutation_chance):
        super().__init__(name, current_value, start_range, end_range, mutation_chance)

    def mutate(self):
        mutation_report = MutationReport()

        if random.random() > self.mutation_chance:
            return mutation_report

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

        mutation_report.attribute_mutations.append(
            self.name + " changed from " + repr(self.current_value) + " to " + repr(new_current_value))

        self.current_value = self.start_range + ((new_current_value - self.start_range) % range)

        return mutation_report

    def _interpolate(self, other):
        return IntegerVariable(self.name, start_range=self.start_range, end_range=self.end_range,
                               current_value=
                               int(round(self.get_current_value() / 2.0 + other.get_current_value() / 2.0)),
                               mutation_chance=self.mutation_chance)
