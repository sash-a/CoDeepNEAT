class Config:
    def __init__(self):
        print('loading config')
        self.fitness_aggregation = 'avg'  # max

        # ------------------------------------------------- NEAT stuff -------------------------------------------------
        # Used when calculating distance between genomes
        self.disjoint_coefficient = 3
        self.excess_coefficient = 5
        # Speciation
        self.n_elite = 1
        self.reproduce_percent = 0.5
        # -------------------------------------------------------------------------------------------------------------


config = Config()
