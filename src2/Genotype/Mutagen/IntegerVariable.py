import random

import math

from src2.Genotype.Mutagen.Variable import Variable
from src2.Genotype.NEAT.Operators.Mutators.MutationReport import MutationReport


class IntegerVariable(Variable):

    def __init__(self, name, current_value: int, start_range: int, end_range: int, mutation_chance):
        super().__init__(name, current_value, start_range, end_range, mutation_chance)
        if current_value % 1 != 0:
            raise Exception("cannot pass non natural number to int variable")

    def mutate(self) -> MutationReport:
        mutation_report = MutationReport()

        if random.random() > self.mutation_chance:
            return mutation_report

        range = self.end_range - self.start_range

        if random.random() < 0.25:
            # random reset
            new_current_value = random.randint(self.start_range, self.end_range)
            change_type = " changed"

        else:
            # random deviation
            deviation = random.normalvariate(0,range/8)
            new_current_value = self.current_value + int(deviation)
            change_type = " deviated"

        # making sure value changes
        if new_current_value == self.current_value:
            new_current_value = self.current_value + random.choice([0, 1])

        new_current_value = self.start_range + ((new_current_value - self.start_range) % range)
        if new_current_value % 1 != 0:
            raise Exception("non natural number mutated in int variable")

        mutation_report += self.name + change_type + " from " + repr(self.current_value) + " to " + repr(
            new_current_value)
        self.current_value = new_current_value

        # print("returning from int var mutagen: ", mutation_report)

        return mutation_report

    def _interpolate(self, other):
        return IntegerVariable(self.name, start_range=self.start_range, end_range=self.end_range,
                               current_value=
                               int(round(self.get_current_value() / 2.0 + other.get_current_value() / 2.0)),
                               mutation_chance=self.mutation_chance)
