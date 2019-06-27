from enum import Enum


class ValueType(Enum):  # TODO assign node type
    DISCRETE = 0
    WHOLE_NUMBERS = 1
    CONTINUOUS = 2


class Mutagen():

    def __init__(self, *discreet_options, current_value=-1, start_range=None, end_range=None,
                 value_type=ValueType.DISCRETE, sub_mutagens: dict = None):
        """defaults to discrete values. can hold whole numbers/ real numbers in a range"""

        self.value_type = value_type

        if (len(discreet_options) > 0):
            self.possible_values = discreet_options
            self.current_value_id = current_value
            #print("possible values:", discreet_options)

        elif (not (start_range is None) and not (end_range is None)):
            if(start_range > current_value or end_range < current_value):
                print("warning: setting current value (",current_value,") of a mutagen to outside the range(",start_range,":",end_range,")")
            self.start_range = start_range
            self.end_range = end_range
            self.current_value = current_value

        else:
            print(
                "error in initialising mutagen. value must either be discreet and provided with options. or numerical values with a provided range")

        self.sub_values= sub_mutagens

    def get_value(self):
        """returns the number value, or the option at curent_value_id
            depending on numerical or discreet mutagen respectively"""
        if (self.value_type == ValueType.DISCRETE):
            return self.possible_values[self.current_value_id]
        else:
            return self.current_value

    def get_sub_value(self, sub_value_name, value = None):
        if value is None:
            return self.sub_values[self.get_value()][sub_value_name].get_value()
        else:
            return self.sub_values[value][sub_value_name].get_value()

    def set_value(self, value):
        """sets current_value=value, or curent_value_id = index(value)
            depending on numerical or discreet mutagen respectively"""
        if (self.value_type == ValueType.DISCRETE):
            self.current_value_id = self.possible_values.index(value)
        else:
            self.current_value = value

    def set_sub_value(self, sub_value_name, sub_value, value = None):

        if value is None:
            self.sub_values[self.get_value()][sub_value_name].set_value(sub_value)
        else:
            self.sub_values[value][sub_value_name].set_value(sub_value)
