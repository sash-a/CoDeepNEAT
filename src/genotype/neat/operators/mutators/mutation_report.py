from typing import List


class MutationReport:
    """
        represents the results of a mutation.
        can be the result of a single small mutation eg: mutagen value change or an add node
        can be the result of a composite of mutations on a whole genome
        can be the result of a whole populations mutations in a single step or for the full run time
    """

    def __init__(self):
        self.nodes_added: int = 0
        self.connections_enabled: int = 0
        self.connections_disabled: int = 0
        self.connections_created: int = 0

        self.attribute_mutations: List[str] = []

    def check_mutated(self):
        return self.nodes_added > 0 or self.connections_enabled > 0 \
               or self.connections_disabled > 0 or self.connections_created > 0 \
               or len(self.attribute_mutations) > 0

    def __str__(self):
        out = ""
        if self.nodes_added > 0:
            out += "nodes added: " + str(self.nodes_added) + "\n"
        if self.connections_created > 0:
            out += "connections created: " + str(self.connections_created) + "\n"
        if self.connections_enabled > 0:
            out += "connections enabled: " + str(self.connections_enabled) + "\n"
        if self.connections_disabled > 0:
            out += "connections disabled: " + str(self.connections_disabled) + "\n"

        for att in self.attribute_mutations:
            out += att + "\n"

        if len(out) == 0:
            return "no mutations"

        return out

    def __add__(self, other):
        if type(other) == str:
            result = MutationReport()
            result.attribute_mutations.append(other)
            return result + self

        if type(other) == type(self):
            result = MutationReport()
            a = self.nodes_added
            b = other.nodes_added
            c = result.nodes_added
            result.nodes_added += self.nodes_added + other.nodes_added
            result.connections_enabled += self.connections_enabled + other.connections_enabled
            result.connections_disabled += self.connections_disabled + other.connections_disabled
            result.connections_created += self.connections_created + other.connections_created

            result.attribute_mutations.extend(self.attribute_mutations)
            result.attribute_mutations.extend(other.attribute_mutations)

            return result
