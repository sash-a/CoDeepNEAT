import math
import random
from enum import Enum


class ValueType(Enum):
    """discrete mutagens mutate between a set of options"""
    DISCRETE = 0
    """whole/continuous numbers mutate a numerical value inside of a range"""
    WHOLE_NUMBERS = 1
    CONTINUOUS = 2


class Mutagen:

    """this class represents any mutatable parameter used by cdn.
        generally speaking mutagens represent gene attributes

        a submutagen is a mutagen which only applies to a specific discrete option of the parent mutagen
        where layer_type is a mutagen, convolutional_window_size is a submutagen of layertype for the value convulutional_layer
    """

    def __init__(self, *discreet_options, name="", current_value=-1, start_range=None, end_range=None,
                 value_type=ValueType.DISCRETE, sub_mutagens: dict = None, discreet_value=None, mutation_chance=None,
                 print_when_mutating=False, distance_weighting=0, inherit_as_discrete=False):
        """defaults to discrete values. can hold whole numbers/ real numbers in a range"""

        self.value_type = value_type
        self.end_range = end_range
        self.start_range = None
        self.print_when_mutating = print_when_mutating
        self.name = name
        self.age = 0
        self.distance_weighting = distance_weighting
        # some whole number values such as species number should not be interpolated during inheritance,
        # as this does not make sense
        self.inherit_as_discrete = inherit_as_discrete

        if len(discreet_options) > 0:
            self.possible_values = discreet_options
            self.current_value_id = current_value

        elif not (start_range is None) and not (end_range is None):
            if start_range > current_value or self.end_range < current_value:
                print("warning: setting current value (", current_value, ") of a mutagen to outside the range(",
                      start_range, ":", self.end_range, ")")
            self.start_range = start_range
            self.current_value = current_value

        else:
            print("error in initialising mutagen. "
                  "value must either be discreet and provided with options. or numerical values with a provided range")

        self.sub_values = sub_mutagens  #maps value:{sub_mutagen_name:submutagen}
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

    value = property(lambda self: self.get_value())

    def inherit(self, other):
        """used by the mutagen breeding extension
            sets this mutagens values and subvalues closer to the values and subvalues of other
            this is the interpolation of this node towards other
        """
        if self.value_type != other.value_type:
            raise Exception("cannot breed mutagens of differing types:", self.value_type, other.value_type)

        if self.value_type == ValueType.DISCRETE or self.inherit_as_discrete:
            """chance to take others value"""
            if random.random() < 0.35:
                self.set_value(other.get_value())
                # print(self.name, "inheritting discrete value",other.get_value())

        else:
            """new value interpolated from old value - skewed slightly towards self against other"""
            my_value = self.get_value()
            other_value = other.get_value()
            new_value = my_value + 0.35 * (other_value - my_value)
            if self.value_type == ValueType.WHOLE_NUMBERS:
                new_value += random.random() * 0.2 - 0.1  # to make it equally likely to round up/down from  x.5
                self.set_value(int(round(new_value)))
            else:
                self.set_value(new_value)

        if not (self.sub_values is None):
            for val in self.sub_values.keys():
                my_subs = self.sub_values[val]
                if val == self.get_value():
                    for sub_mut_name in my_subs.keys():
                        if val in other.sub_values:
                            other_subs = other.sub_values[val]
                            my_subs[sub_mut_name].inherit(other_subs[sub_mut_name])

    def __call__(self):
        return self.get_value()

    def mutate(self, magnitude=1):
        """varies this mutagens values and subvalues
        :returns whether or not this mutagen mutated"""
        old_value = self()
        self.mutate_sub_mutagens()
        if self.print_when_mutating:
            # print("trying to mutate mutagen",self.name,"mutation chance:",self.mutation_chance)
            pass

        self.age += 1

        if random.random() < self.mutation_chance * magnitude:
            if self.value_type == ValueType.DISCRETE:
                new_current_value_id = random.randint(0, len(self.possible_values) - 1)
                if new_current_value_id == self.current_value_id:
                    new_current_value_id = (self.current_value_id + 1) % len(self.possible_values)
                self.current_value_id = new_current_value_id

            if self.value_type == ValueType.WHOLE_NUMBERS:
                if random.random() < 0.25:
                    """random reset"""
                    new_current_value = random.randint(self.start_range, self.end_range)
                else:
                    deviation_fraction = math.pow(random.random(), 4) * (1 if random.random() < 0.5 else -1) * magnitude
                    new_current_value = self.current_value + int(
                        deviation_fraction * (self.end_range - self.start_range))

                if new_current_value == self.current_value:
                    new_current_value = self.current_value + (1 if random.random() < 0.5 else -1)

                new_current_value = max(self.start_range, min(self.end_range - 1, new_current_value))
                self.current_value = new_current_value

            if self.value_type == ValueType.CONTINUOUS:
                if random.random() < 0.25:
                    """random reset"""
                    new_current_value = random.uniform(self.start_range, self.end_range)
                    deviation_fraction = -1
                else:
                    deviation_fraction = math.pow(random.random(), 4) * (1 if random.random() < 0.5 else -1) * magnitude
                    new_current_value = self.current_value + deviation_fraction * (self.end_range - self.start_range)
                new_current_value = max(self.start_range, min(self.end_range, new_current_value))
                if self.print_when_mutating:
                    print("altering continuous number from", old_value, "to", new_current_value, "using dev frac=",
                          deviation_fraction, "range: [", self.start_range, ",", self.end_range, ")")
                self.current_value = new_current_value

            if self.print_when_mutating and old_value != self():
                print("mutated gene from", old_value, "to", self(), "range: ", self.start_range, ", ", self.end_range)

            return not old_value == self()

    def mutate_sub_mutagens(self):
        """part of the recursive mutation"""
        if not (self.sub_values is None):
            for val in self.sub_values.keys():
                subs = self.sub_values[val]
                if val == self.get_value():
                    for sub_mut in subs.values():
                        sub_mut.mutate()

    def get_value(self):
        """returns the number value, or the option at curent_value_id
            depending on numerical or discreet mutagen respectively"""
        if self.value_type == ValueType.DISCRETE:
            return self.possible_values[self.current_value_id]
        else:
            return self.current_value

    def get_sub_value(self, sub_value_name, value=None, return_mutagen=False):
        """returns the submutagen or its value given its name"""
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

    def get_all_sub_values(self):
        sub_values = []
        if self.sub_values is None:
            return sub_values

        for val in self.sub_values.keys():
            if val != self.get_value():
                continue

            subs = self.sub_values[val]
            for sub_mut in subs.values():
                if sub_mut.value_type == ValueType.DISCRETE:
                    sub_values.extend(sub_mut.get_all_sub_values())
                    continue
                sub_values.append(sub_mut.get_value())

        return sub_values

    def __repr__(self):
        return str(self.value_type) + ' ' + str(self.start_range) + ' ' + str(self.end_range)

    def distance_to(self, other):
        """used for attribute distance.
            calculates the distance or similarity between self and an other mutagen of the same kind
        """
        if self.value_type == ValueType.DISCRETE:
            dist = 0
            if self() != other():
                dist = self.distance_weighting
        else:
            dist = self.distance_weighting * abs(self() - other()) / (self.end_range - self.start_range)

        if self.sub_values is None:
            return dist

        for sub_mutagen_group in self.sub_values.keys():
            self_subs = self.sub_values[sub_mutagen_group]
            other_subs = other.sub_values[sub_mutagen_group]
            for sub_mut_key in self_subs.keys():
                dist += self_subs[sub_mut_key].distance_to(other_subs[sub_mut_key])

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
