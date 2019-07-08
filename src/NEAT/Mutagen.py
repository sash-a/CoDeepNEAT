from enum import Enum
import random
import math

class ValueType(Enum):  # TODO assign node type
    DISCRETE = 0
    WHOLE_NUMBERS = 1
    CONTINUOUS = 2


class Mutagen():

    def __init__(self, *discreet_options, current_value=-1, start_range=None, end_range=None,
                 value_type=ValueType.DISCRETE, sub_mutagens: dict = None, discreet_value=None, mutation_chance = None, print_when_mutating = False):
        """defaults to discrete values. can hold whole numbers/ real numbers in a range"""

        self.value_type = value_type
        self._end_range = end_range
        self.print_when_mutating = print_when_mutating

        if (len(discreet_options) > 0):
            self.possible_values = discreet_options
            self.current_value_id = current_value
            # print("possible values:", discreet_options)

        elif (not (start_range is None) and not (end_range is None)):
            if (start_range > current_value or self.end_range < current_value):
                print("warning: setting current value (", current_value, ") of a mutagen to outside the range(",
                      start_range, ":", self.end_range, ")")
            self.start_range = start_range
            self.current_value = current_value

        else:
            print("error in initialising mutagen. "
                  "value must either be discreet and provided with options. or numerical values with a provided range")

        self.sub_values = sub_mutagens
        if not (discreet_value is None):
            self.set_value(discreet_value)

        if(mutation_chance is None):
            if (value_type == ValueType.DISCRETE):
                self.mutation_chance = 0.05
            if (value_type == ValueType.WHOLE_NUMBERS):
                self.mutation_chance = 0.1
            if (value_type == ValueType.CONTINUOUS):
                self.mutation_chance = 0.2
        else:
            self.mutation_chance = mutation_chance
    def __call__(self):
        return self.get_value()

    @property
    def end_range(self):
        if(isinstance(self._end_range, int) or  isinstance(self._end_range, float)):
            return self._end_range
        else:
            print("using function",self._end_range,"to aquire end of range, found:",self._end_range())
            return self._end_range()

    def mutate(self):

        if(random.random()<self.mutation_chance):
            if (self.value_type == ValueType.DISCRETE):
                new_current_value_id = random.randint(0,len(self.possible_values)-1)
                if(new_current_value_id == self.current_value_id):
                    new_current_value_id = (self.current_value_id+1)%len(self.possible_values)
                #print("mutating", self.get_value(), "from id", self.current_value_id,"to",new_current_value_id,"poss=",self.possible_values)
                #print("mutating value from",self.get_value(),"to",self.possible_values[new_current_value_id])
                self.current_value_id = new_current_value_id

            if (self.value_type == ValueType.WHOLE_NUMBERS):
                if(random.random()< 0.2):
                    """random reset"""
                    new_current_value= random.randint(self.start_range, self.end_range)
                else:
                    new_current_value = self.current_value + int((math.pow(random.random(),3)-0.5)*(  self.end_range - self.start_range))

                if (new_current_value == self.current_value):
                    new_current_value = (self.current_value + (1 if random.random()<0.5 else -1) - self.start_range) % (self.end_range - self.start_range) + self.start_range
                    self.current_value = new_current_value
                if (self.print_when_mutating):
                    print("mutated value to:", self.current_value, "range:",self.start_range, ",",self.end_range)

            if (self.value_type == ValueType.CONTINUOUS):
                if (random.random() < 0.1):
                    """random reset"""
                    new_current_value = random.random(self.start_range, self.end_range)
                else:
                    new_current_value = self.current_value + (math.pow(random.random(), 3) - 0.5) * (self.end_range - self.start_range)
                self.current_value = new_current_value

            self.mutate_sub_mutagens()

    def mutate_sub_mutagens(self):
        #print("called mutate_sub_mutagens",self.sub_values)
        if not (self.sub_values is None):
            #print("trying to mutate subs")
            for val in self.sub_values.keys():
                subs = self.sub_values[val]
                #print("trying to mutate sub mut, my val=",self.get_value(), "subs key value=",val)
                if val == self.get_value():
                    for sub_mut in subs.values():
                        #print("mutating submutagen")
                        sub_mut.mutate()

    def get_value(self):
        """returns the number value, or the option at curent_value_id
            depending on numerical or discreet mutagen respectively"""
        if (self.value_type == ValueType.DISCRETE):
            return self.possible_values[self.current_value_id]
        else:
            return self.current_value

    def get_sub_value(self, sub_value_name, value=None, return_mutagen = False):

        if value is None:
            mutagen =  self.sub_values[self.get_value()][sub_value_name]
        else:
            mutagen =  self.sub_values[value][sub_value_name]

        if return_mutagen:
            return mutagen
        else:
            return mutagen.get_value()

    def set_value(self, value):
        """sets current_value=value, or curent_value_id = index(value)
            depending on numerical or discreet mutagen respectively"""
        if (self.value_type == ValueType.DISCRETE):
            self.current_value_id = self.possible_values.index(value)
        else:
            self.current_value = value

    def set_sub_value(self, sub_value_name, sub_value, value=None):

        if value is None:
            self.sub_values[self.get_value()][sub_value_name].set_value(sub_value)
        else:
            self.sub_values[value][sub_value_name].set_value(sub_value)
