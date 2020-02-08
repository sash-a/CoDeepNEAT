import json
import os
from multiprocessing import current_process
from typing import Dict, Optional

from torch import device


class Config:
    def __init__(self):
        # ----------------------------------------------- General stuff -----------------------------------------------
        self.run_name = 'test'
        self.n_generations = 30
        # ------------------------------------------------ Model stuff ------------------------------------------------
        self.device = 'gpu'  # cpu
        self.n_gpus = 1
        self.n_evals_per_gpu = 1
        self.batch_size = 256
        self.epochs_in_evolution = 8
        self.n_evals_per_bp = 4
        self.max_model_params = 50e6

        self.min_square_dim = -1  # Min output size a conv can be without padding
        # --------------------------------------------- Fully train stuff ---------------------------------------------
        self.fully_train = False
        self.resume_fully_train = False  # used to know if a generation should be downloaded from wandb or a fully train should be downloaded
        self.fully_train_accuracy_test_period = 10
        self.fully_train_epochs = 100
        self.fully_train_feature_multiplier = 1
        # ---------------------------------------------- Debug Options ----------------------------------------------
        self.dummy_run = False
        self.dummy_time = 0  # number of seconds to wait to return a dummy eval
        self.max_batches = -1
        # -------------------------------------------- Visualising Options --------------------------------------------
        self.view_graph_plots = False  # if true, any plotted graphs will be viewed
        self.plot_best_genotypes = True
        self.plot_every_genotype = False
        self.plot_best_phenotype = True
        self.plot_every_phenotype = False
        self.plot_module_species = False
        self.view_batch_image = False
        # ----------------------------------------------- Dataset stuff -----------------------------------------------
        self.dataset = 'cifar10'  # mnist | cifar10 | custom
        self.custom_dataset_root = ''
        self.validation_split = 0.15  # Percent of the train set that becomes the validation set
        self.download_dataset = True
        # ------------------------------------------------- DA stuff --------------------------------------------------
        self.evolve_da = False
        self.evolve_da_pop = False
        self.use_colour_augmentations = True
        self.add_da_node_chance = 0.15
        self.apply_da_chance = 0.5
        self.da_link_forget_chance = 0.25
        self.batch_augmentation = True
        # ------------------------------------------------- cdn stuff -------------------------------------------------
        self.multiobjective = False
        # Population and species sizes
        self.module_pop_size = 50
        self.bp_pop_size = 20
        self.da_pop_size = 5

        self.n_module_species = 4
        self.n_blueprint_species = 1
        # Features chances
        self.module_node_batchnorm_chance = 0.65
        self.module_node_dropout_chance = 0.2
        self.module_node_max_pool_chance = 0.3
        # chance of a new node starting with a deep layer - as opposed to a regulariser only layer
        self.module_node_deep_layer_chance = 1
        self.module_node_conv_layer_chance = 1  # chance of linear = 1-conv. not used if no deep layer
        self.lossy_chance = 0.5
        self.mutate_lossy_values = True
        # Layer types
        self.use_depthwise_separable_convs = False
        # Module retention/elitism
        self.fitness_aggregation = 'avg'  # max | avg
        self.use_module_retention = False
        self.module_map_forget_mutation_chance = 0.2
        self.max_module_map_ignores = 1
        self.parent_selector = "uniform"  # uniform | roulette | tournament
        self.representative_selector = 'random'  # best | centroid | random
        # blank node settings - if true input/output nodes are left blank perpetually
        self.blank_module_input_nodes = False
        self.blank_bp_input_nodes = False
        self.blank_module_output_nodes = False
        self.blank_bp_output_nodes = False
        # ------------------------------------------------- neat stuff -------------------------------------------------
        # Used when calculating distance between genomes
        self.disjoint_coefficient = 3
        self.excess_coefficient = 5
        # Speciation
        self.module_speciation = 'neat'  # similar | neat
        self.elite_percent = 0.1
        self.reproduce_percent = 0.2  # Percent of species members that are allowed to reproduce
        # used for neat speciation
        self.species_distance_thresh_mod_base = 1
        self.species_distance_thresh_mod_min = 0.001
        self.species_distance_thresh_mod_max = 100
        # Mutation chances
        self.blueprint_add_node_chance = 0.16  # 0.16
        self.blueprint_add_connection_chance = 0.12  # 0.12
        self.blueprint_node_species_switch_chance = 0.15  # chance per bp
        self.module_add_node_chance = 0.1  # 0.08
        self.module_add_connection_chance = 0.1  # 0.08
        self.module_node_layer_type_change_chance = 0.1
        self.gene_breeding_chance = 0
        self.blueprint_node_type_switch_chance = 0  # 0.1     chance for blueprint nodes to switch to module nodes
        self.max_module_repeats = 1
        self.module_repeat_mutation_chance = 0
        # ------------------------------------------------ wandb stuff ------------------------------------------------
        self.use_wandb = True
        self.wandb_tags = []
        self.wandb_run_path = ''
        # -------------------------------------------------------------------------------------------------------------

    def get_device(self):
        """Used to obtain the correct device taking into account multiple GPUs"""
        gpu = 'cuda:'
        gpu_idx = '0' if not current_process().name.isdigit() else str(int(current_process().name) % self.n_gpus)
        # print('extracted device id:', gpu_idx)
        gpu += gpu_idx
        return device('cpu') if self.device == 'cpu' else device(gpu)

    @staticmethod
    def _build_file_path(file: str) -> str:
        # If the path is not absolute (i.e starts at root) then search in configs dir
        if not file.endswith('.json'):
            file += ".json"

        if not file.startswith("/"):
            file = os.path.join(os.path.dirname(__file__), 'configs', file)

        return file

    def read_option(self, file: str, option: str) -> Optional[any]:
        file = Config._build_file_path(file)

        with open(file) as cfg_file:
            options: dict = json.load(cfg_file)
            if option in options and option in self.__dict__:
                return options[option]

        return None

    def read(self, file: str):
        file = Config._build_file_path(file)

        with open(file) as cfg_file:
            options: dict = json.load(cfg_file)
            # print(file, options)
            self._add_cfg_dict(options)

    def _add_cfg_dict(self, options: Dict[str, any]):
        self._load_inner_configs(options)

        for option_name, option_value in options.items():
            if isinstance(option_value, dict):  # If an option value is a dict, then check the dict for sub options
                self._add_cfg_dict(option_value)
                continue
            if option_name in self.__dict__:  # Only add an option if it has exactly the same name as a variable
                if option_name == "wandb_tags":
                    self.__dict__[option_name].extend(option_value)
                else:
                    self.__dict__[option_name] = option_value

    def _load_inner_configs(self, options: Dict[str, any]):
        inner_configs_key = 'configs'
        if inner_configs_key in options:
            # has inner configs
            inner_configs = options[inner_configs_key]
            if isinstance(inner_configs, dict):
                for config_name in reversed(list(inner_configs.keys())):
                    if inner_configs[config_name]:
                        print("reading inner config:",config_name)
                        self.read(config_name)
            else:
                raise TypeError('Expected a list of other config options, received: ' + str(type(inner_configs)))
