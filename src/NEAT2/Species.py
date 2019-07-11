class Species:
    def __init__(self):
        # Possible extra attribs:
        # age, hasBest, noImprovement
        self.id

        self.representative
        self.members

        self.num_children

    def __iter__(self):
        return iter(self.members)

    def __len__(self):
        return len(self.members)

    def add(self):
        pass

    def calc_num_children(self):
        pass

    def reproduce(self):
        pass
