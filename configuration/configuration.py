import json
import os
from multiprocessing import current_process
from typing import Dict, Optional

from torch import device


class Config:
    def __init__(self):
        # ----------------------------------------------- General stuff -----------------------------------------------
        self.run_name = 'test'
        self.batch_run_scheduler = ''
        self.n_generations = 50
        # ------------------------------------------------ Timing stuff ------------------------------------------------
        self.allowed_runtime_sec = -1  # Amount of time allowed for the run (-1 for infinite)
        # ------------------------------------------ General Evaluation stuff -----------------------------------------
        self.loss_based_stopping_in_evolution = False
        self.loss_based_stopping_max_epochs = 20
        self.loss_gradient_threshold = 0.012
        self.optim = 'adam'  # sgd | adam | evolve
        self.batch_size = 256
        self.n_evals_per_bp = 4
        # ------------------------------------------------ Model stuff ------------------------------------------------
        self.device = 'gpu'  # cpu
        self.n_gpus = 1
        self.n_evals_per_gpu = 1
        self.epochs_in_evolution = 8
        self.max_model_params = 50e6

        self.min_square_dim = -1  # Min output size a conv can be without padding
        # --------------------------------------------- Fully train stuff ---------------------------------------------
        self.fully_train = False
        self.resume_fully_train = False  # used to know if a generation should be downloaded from wandb or a fully train should be downloaded

        self.fully_train_max_epochs = 150
        self.fully_train_best_n_blueprints = 5

        self.ft_feature_multipliers = [1, 3, 5]

        self.fully_train_accuracy_test_period = 10
        self.ft_retries = True  # retry if accuracy seems too low
        self.ft_auto_stop_count = -1  # number of acc plateaus before a stop. -1 for no auto stopping

        self.ft_allow_lr_drops = True  # drops the learning rate if accuracy plateaus
        self.lr_drop_fac = 2
        # ---------------------------------------------- Debug Options ----------------------------------------------
        self.dummy_run = False
        self.dummy_time = 0  # number of seconds to wait to return a dummy eval
        self.max_batches = -1  # if not -1, will stop all epoch at specified max batches
        # -------------------------------------------- Visualising Options --------------------------------------------
        self.view_graph_plots = False  # if true, any plotted graphs will be viewed
        self.plot_best_genotypes = True
        self.plot_every_genotype = False
        self.plot_best_phenotype = True
        self.plot_every_phenotype = False
        self.plot_module_species = False
        self.view_batch_image = False
        self.visualise_moo_scores = False
        # ----------------------------------------------- Dataset stuff -----------------------------------------------
        self.dataset = 'cifar10'  # mnist | cifar10 | custom
        self.custom_dataset_root = ''
        self.validation_split = 0.15  # Percent of the train set that becomes the validation set
        self.download_dataset = True
        # ------------------------------------------------- DA stuff --------------------------------------------------
        self.evolve_da = True
        self.evolve_da_pop = False
        self.use_colour_augmentations = True
        self.add_da_node_chance = 0.15
        self.da_op_swap_chance = 0.25
        self.apply_da_chance = 0.5
        self.da_link_forget_chance = 0.25
        self.batch_augmentation = True
        # ------------------------------------------------- cdn stuff -------------------------------------------------
        self.multiobjective = True
        # Population and species sizes
        self.module_pop_size = 50
        self.bp_pop_size = 20
        self.da_pop_size = 20

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
        self.element_wise_multiplication_chance = 0
        # Layer types
        self.use_depthwise_separable_convs = False
        # Module retention/elitism
        self.fitness_aggregation = 'avg'  # max | avg
        self.use_module_retention = False
        self.module_map_forget_mutation_chance = 0.2  # chance for a blueprint to forget a linked module during mutation
        self.max_module_map_ignores = -1  # max number of eval ignores (eval 0 has no ignores) | -1 = unbounded
        self.parent_selector = "roulette"  # uniform | roulette | tournament
        self.tournament_size = 5
        self.representative_selector = 'random'  # best | centroid | random
        # blank node settings - if true input/output nodes are left blank perpetually
        self.blank_module_input_nodes = False
        self.blank_bp_input_nodes = False
        self.blank_module_output_nodes = False
        self.blank_bp_output_nodes = False
        # The starting shapes for nodes of both population: io_only | linear | triangle | diamond
        self.initial_shapes = ['io_only']  # {'io_only', 'linear', 'triangle', 'diamond'}
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
        self.max_layer_repeats = 1
        self.module_repeat_mutation_chance = 0
        self.layer_repeat_mutation_chance = 0
        self.target_network_size = -1  # if this option is used, all phenotype net sizes will be adjusted to match this as close as possible
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
    def build_file_path(file: str) -> str:
        # If the path is not absolute (i.e starts at root) then search in configs dir
        if not file.endswith('.json'):
            file += ".json"

        if not file.startswith("/"):
            # a config can be specified with a full path, allowing it to be saved outside the configs folder
            # prepend the full path up to the configs folder
            preceding_path = Config.find_path_containing_file(file)
            file = os.path.join(preceding_path, file)

        return file

    @staticmethod
    def find_path_containing_file(file):
        bare_file = file.split("/")[-1]  # just conf_name.json
        path_items = file.split("/")[:-1]

        configs_folder = os.path.join(os.path.dirname(__file__), 'configs')
        # fetch all dir which contain the bare_file name
        contained_dirs = [sub[0] for sub in os.walk(configs_folder) if bare_file in sub[2]]
        # filter leaving ones which end in the given path
        if len(path_items) > 0:
            # a path was specified, find the containing dirs which match this path at their ends
            contained_dirs = [d for d in contained_dirs if path_items == d.split("/")[-len(path_items):]]

        if len(contained_dirs) > 1:
            raise Exception("ambiguous config specified. given " + file + " found in " + repr(contained_dirs))
        if len(contained_dirs) == 0:
            raise Exception("cannot find specified config " + file + " in " + configs_folder)

        return contained_dirs[0]

    def read_option(self, file: str, option: str) -> Optional[any]:
        file = Config.build_file_path(file)

        with open(file) as cfg_file:
            options: dict = json.load(cfg_file)
            if option in options and option in self.__dict__:
                return options[option]

        return None

    def read(self, file: str):
        file = Config.build_file_path(file)
        print('Reading config file with path {}'.format(file))

        with open(file) as cfg_file:
            options: dict = json.load(cfg_file)
            # print(file, options)
            self._add_cfg_dict(options)

        tags = sorted(list(set(self.__dict__["wandb_tags"])))  # unique in order
        self.__dict__["wandb_tags"] = tags

    def _add_cfg_dict(self, options: Dict[str, any]):
        self._load_inner_configs(options)

        for option_name, option_value in options.items():
            if isinstance(option_value, dict):  # If an option value is a dict, then check the dict for sub options
                self._add_cfg_dict(option_value)
                continue
            if option_name in self.__dict__:  # Only add an option if it has exactly the same name as a variable
                if option_name == "wandb_tags":
                    tags = self.__dict__["wandb_tags"]
                    tags.extend(option_value)
                    tags = sorted(list(set(tags)))  # unique in order
                    self.__dict__["wandb_tags"] = tags
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
                        print("reading inner config:", config_name)
                        self.read(config_name)
            else:
                raise TypeError('Expected a list of other config options, received: ' + str(type(inner_configs)))
