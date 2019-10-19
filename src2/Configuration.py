class Config:
    def __init__(self):
        print('loading config')
        self.fitness_aggregation = 'avg'  # max

        # NEAT stuff
        self.disjoint_coefficient = 3
        self.excess_coefficient = 5

config = Config()
