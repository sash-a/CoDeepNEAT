import random
from typing import Dict, Any

from src2.Genotype.Mutagen.Mutagen import Mutagen


class Option(Mutagen):

    def __init__(self, name: str, *options, current_value=None,
                 submutagens: Dict[Any, Dict[str, Mutagen]] = None,
                 mutation_chance: float = 0.3):

        if current_value is None:
            raise Exception("must provide a current value")

        super(name, mutation_chance)
        self.options = options

        """maps an option value to -> a mapping from subvalue name to -> submutagen"""
        self.submutagens: Dict[Any, Dict[str, Mutagen]] = submutagens

        if current_value not in options:
            raise Exception("current value must be in options list")
        self.current_value = current_value

    def get_subvalue(self, subvalue_name):
        return self.get_submutagen(subvalue_name).value

    def get_submutagen(self, subvalue_name):
        return self.submutagens[self.value][subvalue_name]

    def get_submutagens(self):
        return self.submutagens[self.value].values()

    def get_current_value(self):
        return self.current_value

    def mutate(self):
        if random.random() < self.mutation_chance:
            if (len(self.options) < 2):
                raise Exception("too few options to mutate")

            new_value = self.options[random.randint(0, len(self.options) - 1)]

            while (new_value == self()):
                new_value = self.options[random.randint(0, len(self.options) - 1)]

            self.current_value = new_value

        self.mutate_sub_mutagens()

    def mutate_sub_mutagens(self):
        for sub in self.get_submutagens():
            sub.mutate()
