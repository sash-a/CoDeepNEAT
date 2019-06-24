compat_thresh = 3


class Species:
    def __init__(self, representative):
        self.representative = representative
        self.members = [representative]

    def add_member(self, new_member, thresh=compat_thresh):
        if self.is_compatible(new_member, thresh):
            self.members.append(new_member)

    def is_compatible(self, individual, thresh=compat_thresh, c1=1, c2=1):
        n = max(len(self.representative.connections), len(individual.connections))
        self_d, self_e = self.representative.get_disjoint_excess(individual)
        other_d, other_e = individual.get_disjoint_excess(self.representative)

        d = len(self_d) + len(other_d)
        e = len(self_e) + len(other_e)

        compatibility = (c1 * d + c2 * e) / n
        return compatibility <= thresh

    def clear(self):
        self.members.clear()
