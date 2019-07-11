class Population:
    def __init__(self, individuals):
        self.species = []
        self.speciate(individuals)

    individuals = property(lambda self: self.get_all_individuals())

    def __iter__(self):
        return iter(self.get_all_individuals())

    def get_all_individuals(self):
        pass

    def save_checkpoint(self):
        pass

    def load_checkpoint(self):
        pass

    def speciate(self, individuals):
        pass

    def get_num_children(self, species):
        pass

    def step(self):
        pass


print(Population(None).speciate(None))
