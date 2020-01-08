import random
from typing import Dict, Any, Union

from src2.genotype.mutagen import mutagen as MutagenFile
from src2.genotype.mutagen.mutagen import Mutagen
from src2.genotype.neat.operators.mutators.mutation_report import MutationReport


class _Null:
    """Default current value, allows for an option to be None"""
    pass


class Option(Mutagen):

    def __init__(self, name: str, *options, current_value=_Null, submutagens: Dict[Any, Dict[str, Mutagen]] = None,
                 mutation_chance: float = 0.3):
        super().__init__(name, mutation_chance)

        if current_value is _Null or current_value == 'auto':
            current_value = random.choice(options)

        self.options = options

        # maps an option value to -> a mapping from subvalue name to -> submutagen
        self.submutagens: Dict[Any, Dict[str, Union[Mutagen, Option]]] = submutagens

        if current_value not in options:
            raise Exception("Current value must be option in list: " + repr(current_value) + " not in " + repr(options))
        self.current_value = current_value

    def __repr__(self):
        out: str = self.name + ": " + repr(self.current_value)
        if self.submutagens is None or self.value not in self.submutagens:
            return out

        out += "\n"
        subs = self.submutagens[self.value]
        i = 0
        for sub in subs:
            out += repr(subs[sub]) + ("\t" if i % 2 == 0 else "\n")
            i+=1
        return out

    def get_subvalue(self, subvalue_name):
        return self.get_submutagen(subvalue_name).value

    def get_submutagen(self, subvalue_name):
        if self.submutagens is None:
            raise Exception("No submutagens on option: " + repr(self.name) + " " + repr(self))

        if self.value not in self.submutagens:
            print(self.name, "does not have the submutagen", subvalue_name, "does not have any submutagens")

        if subvalue_name not in self.submutagens[self.value]:
            raise Exception(
                self.name + " does not have the submutagen " + subvalue_name + " for value " + repr(self.value))

        return self.submutagens[self.value][subvalue_name]

    def get_submutagens(self):
        if self.submutagens is None:
            return []

        if self.value not in self.submutagens:
            return []

        return self.submutagens[self.value].values()

    def get_current_value(self):
        return self.current_value

    def mutate(self) -> MutationReport:
        mutation_report = MutationReport()
        if random.random() < self.mutation_chance:
            if len(self.options) < 2:
                raise Exception("too few options to mutate")

            new_value = self.options[random.randint(0, len(self.options) - 1)]

            while new_value == self():
                new_value = self.options[random.randint(0, len(self.options) - 1)]

            mutation_report += self.name + " changed from " + repr(self.current_value) + " to " + repr(new_value)
            self.current_value = new_value

        return mutation_report + self.mutate_sub_mutagens()

    def mutate_sub_mutagens(self) -> MutationReport:
        mutation_report = MutationReport()
        for sub in self.get_submutagens():
            mutation_report += sub.mutate()

        return mutation_report

    def set_value(self, value):
        if value not in self.options:
            raise Exception("trying to set the value of the " + self.name + " mutagen to "
                            + repr(value) + " which is not in the options: " + repr(self.options))

        self.current_value = value

    def set_sub_value(self, submutagen_name, value):
        self.get_submutagen(submutagen_name).set_value(value)

    def interpolate(self, other: Mutagen):
        return Option(self.name, *self.options,
                      current_value=random.choice([self.current_value, other.get_current_value()]),
                      submutagens=interpolate_submutagens(self, other))


def interpolate_submutagens(mutagen_a: Option, mutagen_b: Option):
    subs = {}
    if mutagen_a.submutagens is None:
        return subs

    for val in mutagen_a.submutagens.keys():
        subs[val] = {}

        for name in mutagen_a.submutagens[val].keys():
            subs[val][name] = MutagenFile.interpolate(mutagen_a.submutagens[val][name],
                                                      mutagen_b.submutagens[val][name])

    return subs
