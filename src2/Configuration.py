class Config:
    def __init__(self):
        print('loading config')
        # ----------------------------------------------- General stuff -----------------------------------------------
        self.n_gpus = 1
        # ------------------------------------------------- CDN stuff -------------------------------------------------
        # Features chances
        self.module_node_batchnorm_chance = 0.65
        self.module_node_dropout_chance = 0.2
        self.module_node_max_pool_chance = 0.3
        self.module_node_deep_layer_chance = 0.65
        self.module_node_conv_layer_chance = 0.65  # chance of linear = 1-conv. not used if no deep layer
        # Layer types
        self.use_depthwise_separable_convs = True
        #
        self.fitness_aggregation = 'avg'  # max
        #
        self.evaluations = 30  # Number of times to sample a blueprint to evaluate
        # ------------------------------------------------- NEAT stuff -------------------------------------------------
        # Used when calculating distance between genomes
        self.disjoint_coefficient = 3
        self.excess_coefficient = 5
        # Speciation
        self.n_elite = 1
        self.reproduce_percent = 0.5  # Percent of species members that are allowed to reproduce
        self.species_distance_thresh_mod_base = 1
        self.species_distance_thresh_mod_min = 1
        self.species_distance_thresh_mod_max = 1
        # -------------------------------------------------------------------------------------------------------------


config = Config()
