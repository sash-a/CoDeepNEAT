import math
import random
from enum import Enum


class ValueType(Enum):
    DISCRETE = 0
    WHOLE_NUMBERS = 1
    CONTINUOUS = 2


class Mutagen:

    def __init__(self, *discreet_options, name="", current_value=-1, start_range=None, end_range=None,
                 value_type=ValueType.DISCRETE, sub_mutagens: dict = None, discreet_value=None, mutation_chance=None,
                 print_when_mutating=False, distance_weighting=0):
        """defaults to discrete values. can hold whole numbers/ real numbers in a range"""

        self.value_type = value_type
        self.end_range = end_range
        self.start_range = None
        self.print_when_mutating = print_when_mutating
        self.name = name
        self.age = 0
        self.distance_weighting = distance_weighting

        if len(discreet_options) > 0:
            self.possible_values = discreet_options
            self.current_value_id = current_value
            # print("possible values:", discreet_options)

        elif not (start_range is None) and not (end_range is None):
            if start_range > current_value or self.end_range < current_value:
                print("warning: setting current value (", current_value, ") of a mutagen to outside the range(",
                      start_range, ":", self.end_range, ")")
            self.start_range = start_range
            self.current_value = current_value

        else:
            print("error in initialising mutagen. "
                  "value must either be discreet and provided with options. or numerical values with a provided range")

        self.sub_values = sub_mutagens
        if value_type == ValueType.DISCRETE:
            self.set_value(discreet_value)

        if mutation_chance is None:
            if value_type == ValueType.DISCRETE:
                self.mutation_chance = 0.05
            if value_type == ValueType.WHOLE_NUMBERS:
                self.mutation_chance = 0.1
            if value_type == ValueType.CONTINUOUS:
                self.mutation_chance = 0.2
        else:
            self.mutation_chance = mutation_chance

    def __call__(self):
        # print("calling, returning:", self.get_value())
        return self.get_value()

    def mutate(self):
        """:returns whether or not this gene mutated"""
        old_value = self()
        self.mutate_sub_mutagens()
        if self.print_when_mutating:
            # print("trying to mutate mutagen",self.name,"mutation chance:",self.mutation_chance)
            pass

        self.age += 1

        if random.random() < self.mutation_chance:
            if self.value_type == ValueType.DISCRETE:
                new_current_value_id = random.randint(0, len(self.possible_values) - 1)
                if new_current_value_id == self.current_value_id:
                    new_current_value_id = (self.current_value_id + 1) % len(self.possible_values)
                # print("mutating", self.get_value(), "from id", self.current_value_id,"to",new_current_value_id,"poss=",self.possible_values)
                # print("mutating value from",old_value,"to",self.possible_values[new_current_value_id])
                self.current_value_id = new_current_value_id

            if self.value_type == ValueType.WHOLE_NUMBERS:
                if random.random() < 0.2:
                    """random reset"""
                    new_current_value = random.randint(self.start_range, self.end_range)
                else:
                    deviation_fraction = math.pow(random.random(), 4) * (1 if random.random() < 0.5 else -1)
                    new_current_value = self.current_value + int(
                        deviation_fraction * (self.end_range - self.start_range))
                    # print("altering whole number from",old_value,"to",new_current_value,"using dev frac=",deviation_fraction,"range: [",self.start_range,",",self.end_range,")")

                if new_current_value == self.current_value:
                    new_current_value = self.current_value + (1 if random.random() < 0.5 else -1)
                    # print("readjusting value to",new_current_value)

                new_current_value = max(self.start_range, min(self.end_range - 1, new_current_value))
                self.current_value = new_current_value

                # print("mutating whole number from", old_value, "to",self.current_value, "range:",self.start_range,self.end_range)

            if self.value_type == ValueType.CONTINUOUS:
                if random.random() < 0.1:
                    """random reset"""
                    new_current_value = random.uniform(self.start_range, self.end_range)
                    deviation_fraction = -1
                else:
                    deviation_fraction = math.pow(random.random(), 4) * (1 if random.random() < 0.5 else -1)
                    new_current_value = self.current_value + deviation_fraction * (self.end_range - self.start_range)
                new_current_value = max(self.start_range, min(self.end_range, new_current_value))
                if self.print_when_mutating:
                    print("altering continuous number from", old_value, "to", new_current_value, "using dev frac=",
                          deviation_fraction, "range: [", self.start_range, ",", self.end_range, ")")
                self.current_value = new_current_value

            if self.print_when_mutating and old_value != self():
                print("mutated gene from", old_value, "to", self(), "range: [", self.start_range, ",", self.end_range,
                      ")")

            return not old_value == self()

    def mutate_sub_mutagens(self):
        # print("called mutate_sub_mutagens",self.sub_values)
        if not (self.sub_values is None):
            # print("trying to mutate subs")
            for val in self.sub_values.keys():
                subs = self.sub_values[val]
                # print("trying to mutate sub mut, my val=",self.get_value(), "subs key value=",val)
                if val == self.get_value():
                    for sub_mut in subs.values():
                        # print("mutating submutagen",sub_mut())
                        sub_mut.mutate()

    def get_value(self):
        """returns the number value, or the option at curent_value_id
            depending on numerical or discreet mutagen respectively"""
        if self.value_type == ValueType.DISCRETE:
            return self.possible_values[self.current_value_id]
        else:
            # print("returning:",self.current_value)
            return self.current_value

    def get_sub_value(self, sub_value_name, value=None, return_mutagen=False):
        if value is None:
            mutagen = self.sub_values[self.get_value()]
        else:
            mutagen = self.sub_values[value][sub_value_name]
        if sub_value_name in mutagen:
            mutagen = mutagen[sub_value_name]
        else:
            return None

        if return_mutagen:
            return mutagen
        else:
            return mutagen.get_value()

    def get_sub_values(self):
        if not (self.sub_values is None):
            if self.get_value() in self.sub_values:
                return self.sub_values[self.get_value()]

    def distance_to(self, other):
        if self.value_type == ValueType.DISCRETE:
            dist = 0
            if self() != other():
                dist = self.distance_weighting
        else:
            dist = self.distance_weighting * (self() - other()) / (self.end_range - self.start_range)

        if self.sub_values is None:
            return dist

        for sub_value_name, sub_value in self.sub_values.items():
            dist += sub_value.distance_to(other.get_sub_value(sub_value_name, return_mutagen=True))

        return dist

    def set_value(self, value):
        """sets current_value=value, or current_value_id = index(value)
            depending on numerical or discreet mutagen respectively"""
        if self.value_type == ValueType.DISCRETE:
            if value is None and None not in self.possible_values:
                self.current_value_id = 0
            else:
                self.current_value_id = self.possible_values.index(value)
        else:
            self.current_value = value

    def set_sub_value(self, sub_value_name, sub_value, value=None):
        if value is None:
            self.sub_values[self.get_value()][sub_value_name].set_value(sub_value)
        else:
            self.sub_values[value][sub_value_name].set_value(sub_value)
