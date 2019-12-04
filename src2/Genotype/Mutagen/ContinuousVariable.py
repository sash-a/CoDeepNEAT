import random

import math

from src2.Genotype.Mutagen.Variable import Variable
from src2.Genotype.NEAT.Operators.Mutators.MutationReport import MutationReport


class ContinuousVariable(Variable):

    def __init__(self, name, current_value: float, start_range: float, end_range: float, mutation_chance):
        super().__init__(name, current_value, start_range, end_range, mutation_chance)

    def mutate(self) -> MutationReport:
        mutation_report = MutationReport()

        if random.random() > self.mutation_chance:
            return mutation_report

        range = self.end_range - self.start_range

        if random.random() < 0.25:
            # random reset
            new_current_value = random.uniform(self.start_range, self.end_range)
            mutation_report += self.name + " changed from " + repr(self.current_value) + " to " + repr(
                new_current_value)

            self.current_value = new_current_value

        else:
            # random deviation
            deviation = random.normalvariate(0,range/8)
            new_current_value = self.current_value + deviation
            new_current_value = self.start_range + ((new_current_value - self.start_range) % range)

            mutation_report += self.name + " deviated from " + repr(self.current_value) + " to " + repr(
                new_current_value)
            self.current_value = new_current_value

        if mutation_report is None:
            raise Exception("none mutation report in " + self.name)

        # print("returning from cont var mutagen: ", mutation_report)

        return mutation_report

    def _interpolate(self, other):
        return ContinuousVariable(self.name, start_range=self.start_range, end_range=self.end_range,
                                  current_value=self.get_current_value() / 2.0 + other.get_current_value() / 2.0,
                                  mutation_chance=self.mutation_chance)
