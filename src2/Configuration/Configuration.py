import json
from threading import current_thread
from typing import Dict

from torch import device


class Config:
    def __init__(self):
        print('loading config')
        # ----------------------------------------------- General stuff -----------------------------------------------
        self.run_name = 'base'
        self.n_generations = 1000
        # ------------------------------------------------ Model stuff ------------------------------------------------
        self.device = 'gpu'  # cpu
        self.n_gpus = 1
        self.batch_size = 64
        self.epochs_in_evolution = 8
        self.n_evaluations_per_bp = 4
        self.max_model_params = 50e6
        # ---------------------------------------------- Debug Options ----------------------------------------------
        self.dummy_run = True
        self.plot_best_genotypes = False
        self.plt_every_genotype = True

        # ----------------------------------------------- Dataset stuff -----------------------------------------------
        self.dataset = 'cifar10'  # mnist | cifar10 | custom
        self.custom_dataset_root = ''
        self.validation_split = 0.05  # Percent of the train set that becomes the validation set
        self.download_dataset = True
        # ------------------------------------------------- CDN stuff -------------------------------------------------
        self.multiobjective = False
        # Population sizes
        self.module_pop_size = 50
        self.bp_pop_size = 20
        self.da_pop_size = 5

        self.n_module_species = 4
        self.n_blueprint_species = 1

        # Features chances
        self.module_node_batchnorm_chance = 0.65
        self.module_node_dropout_chance = 0.2
        self.module_node_max_pool_chance = 0.3
        self.module_node_deep_layer_chance = 0.95
        self.module_node_conv_layer_chance = 0.7  # chance of linear = 1-conv. not used if no deep layer
        # Layer types
        self.use_depthwise_separable_convs = False
        #
        self.fitness_aggregation = 'avg'  # max | avg
        self.use_module_retention = False
        self.parent_selector = "uniform"  # uniform | roulette | tournament
        self.representative_selector = "random"  # best | centroid | random

        #  blank node settings - if true input/output nodes are left blank perpetually
        self.blank_blueprint_input_nodes = False
        self.blank_blueprint_output_nodes = False
        self.blank_module_input_nodes = False
        self.blank_module_output_nodes = False

        # ------------------------------------------------- NEAT stuff -------------------------------------------------
        # Used when calculating distance between genomes
        self.disjoint_coefficient = 3
        self.excess_coefficient = 5
        # Speciation
        self.module_speciation = "neat"  # similar | neat
        self.n_elite = 1
        self.reproduce_percent = 0.3  # Percent of species members that are allowed to reproduce
        # used for neat speciation
        self.species_distance_thresh_mod_base = 1
        self.species_distance_thresh_mod_min = 0.001
        self.species_distance_thresh_mod_max = 100
        # Mutation chances
        self.blueprint_add_node_chance = 0.16  # 0.16
        self.blueprint_add_connection_chance = 0.12  # 0.12
        self.blueprint_node_type_switch_chance = 0  # 0.1     chance to switch to a module node
        self.blueprint_node_species_switch_chance = 0.15
        self.module_add_node_chance = 0.1  # 0.08
        self.module_add_connection_chance = 0.1  # 0.08
        self.module_node_layer_type_change_chance = 0.1
        # ------------------------------------------------ wandb stuff ------------------------------------------------
        self.use_wandb = False
        # -------------------------------------------------------------------------------------------------------------

    def get_device(self):
        """Used to obtain the correct device taking into account multiple GPUs"""
        gpu = 'cuda:'
        gpu_idx = '0' if current_thread().name == 'MainThread' else str(int(current_thread().name[-1]) % self.n_gpus)
        gpu += gpu_idx
        return device('cpu') if self.device == 'cpu' else device(gpu)

    def read(self, file: str):
        with open(file) as cfg_file:
            options: dict = json.load(cfg_file)
            self._add_cfg_dict(options)

    def _add_cfg_dict(self, options: Dict[str, any]):
        for option_name, option_value in options.items():
            if isinstance(option_value, dict):  # If an option value is a dict, then check the dict for sub options
                self._add_cfg_dict(option_value)
                continue
            if option_name in self.__dict__:  # Only add an option if it has exactly the same name as a variable
                self.__dict__[option_name] = option_value
