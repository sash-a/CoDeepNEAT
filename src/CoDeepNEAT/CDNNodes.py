import copy
import heapq
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.Config import Config, NeatProperties as Props
from src.NEAT.Gene import NodeGene, NodeType
from src.NEAT.Mutagen import Mutagen
from src.NEAT.Mutagen import ValueType

use_convs = True
use_linears = True


class ModulenNEATNode(NodeGene):
    def __init__(self, id, node_type=NodeType.HIDDEN, activation=F.relu, layer_type=nn.Conv2d,
                 conv_window_size=7, conv_stride=1, max_pool_size=2):
        """initialises all of the nodes mutagens, and assigns initial values"""

        super(ModulenNEATNode, self).__init__(id, node_type)

        batch_norm_chance = 0.65  # chance that a new node will start with batch norm
        use_batch_norm = random.random() < batch_norm_chance

        dropout_chance = 0.2  # chance that a new node will start with drop out
        use_dropout = random.random() < dropout_chance

        max_pool_chance = 0.3  # chance that a new node will start with drop out
        use_max_pool = random.random() < max_pool_chance

        self.activation = Mutagen(F.relu, F.leaky_relu, torch.sigmoid, F.relu6,
                                  discreet_value=activation, name="activation function",
                                  mutation_chance=0.15)  # TODO try add in Selu, Elu

        conv_out_features = 25 + random.randint(0, 25)
        linear_out_features = 100 + random.randint(0, 100)

        linear_submutagens = \
            {
                "regularisation": Mutagen(None, nn.BatchNorm1d,
                                          discreet_value=nn.BatchNorm1d if use_batch_norm else None,
                                          mutation_chance=0.15),

                "dropout": Mutagen(None, nn.Dropout, discreet_value=nn.Dropout if use_dropout else None, sub_mutagens=
                {
                    nn.Dropout: {
                        "dropout_factor": Mutagen(value_type=ValueType.CONTINUOUS, current_value=0.15, start_range=0,
                                                  end_range=0.75)}
                }, mutation_chance=0.08),

                "out_features": Mutagen(value_type=ValueType.WHOLE_NUMBERS, current_value=linear_out_features,
                                        start_range=10,
                                        end_range=1024, name="num out features", mutation_chance=0.22,
                                        distance_weighting=Props.LAYER_SIZE_COEFFICIENT if Config.  allow_attribute_distance else 0)
            }

        conv_submutagens = {
            "conv_window_size": Mutagen(3, 5, 7, discreet_value=conv_window_size, mutation_chance=0.13),

            "conv_stride": Mutagen(value_type=ValueType.WHOLE_NUMBERS, current_value=conv_stride, start_range=1,
                                   end_range=5),

            "reduction": Mutagen(None, nn.MaxPool2d, discreet_value=nn.MaxPool2d if use_max_pool else None,
                                 sub_mutagens=
                                 {
                                     nn.MaxPool2d: {"pool_size": Mutagen(
                                         value_type=ValueType.WHOLE_NUMBERS, current_value=max_pool_size, start_range=2,
                                         end_range=5)}
                                 }, mutation_chance=0.15),

            "regularisation": Mutagen(None, nn.BatchNorm2d, discreet_value=nn.BatchNorm2d if use_batch_norm else None,
                                      mutation_chance=0.15),

            "dropout": Mutagen(None, nn.Dropout2d, discreet_value=nn.Dropout2d if use_dropout else None, sub_mutagens=
            {
                nn.Dropout2d: {
                    "dropout_factor": Mutagen(value_type=ValueType.CONTINUOUS, current_value=0.1,
                                              start_range=0, end_range=0.75)}
            }, mutation_chance=0.08),

            "out_features": Mutagen(value_type=ValueType.WHOLE_NUMBERS, current_value=conv_out_features, start_range=1,
                                    end_range=100, name="num out features", mutation_chance=0.22,
                                    distance_weighting=Props.LAYER_SIZE_COEFFICIENT if Config.allow_attribute_distance else 0)
        }

        if use_linears and not use_convs:
            self.layer_type = Mutagen(nn.Linear, discreet_value=nn.Linear,
                                      distance_weighting=Props.LAYER_TYPE_COEFFICIENT if Config.allow_attribute_distance else 0,
                                      sub_mutagens={nn.Linear: linear_submutagens}
                                      )
        if use_convs and not use_linears:
            self.layer_type = Mutagen(nn.Conv2d, discreet_value=nn.Conv2d,
                                      distance_weighting=Props.LAYER_TYPE_COEFFICIENT if Config.allow_attribute_distance else 0,
                                      sub_mutagens={nn.Conv2d: conv_submutagens})
        if use_convs and use_linears:
            self.layer_type = Mutagen(nn.Conv2d, nn.Linear, discreet_value=layer_type,
                                      distance_weighting=Props.LAYER_TYPE_COEFFICIENT if Config.allow_attribute_distance else 0,
                                      sub_mutagens={
                                          nn.Conv2d: conv_submutagens,
                                          nn.Linear: linear_submutagens
                                      }, name="deep layer type", mutation_chance=0.08)

    def get_all_mutagens(self):
        return [self.activation, self.layer_type]

    def __repr__(self):
        return 'Module node:' + str(self.layer_type) + ' ' + str(self.layer_type.get_sub_value('out_features'))

    def get_node_name(self):
        return repr(self.layer_type()) + "\n" + "features: " + repr(self.layer_type.get_sub_value("out_features"))

    def get_complexity(self):
        """approximates the size of this layer in terms of trainable parameters"""
        if self.layer_type() == nn.Conv2d:
            return pow(self.layer_type.get_sub_value("conv_window_size"), 2) * self.layer_type.get_sub_value(
                "out_features")
        elif self.layer_type() == nn.Linear:
            return self.layer_type.get_sub_value("out_features")
        else:
            raise Exception()


class BlueprintNEATNode(NodeGene):
    def __init__(self, id, node_type=NodeType.HIDDEN, representative=None):
        super(BlueprintNEATNode, self).__init__(id, node_type)

        self.species_number = Mutagen(value_type=ValueType.WHOLE_NUMBERS, current_value=0, start_range=0,
                                      end_range=1, print_when_mutating=False, name="species number",
                                      mutation_chance=0.5, inherit_as_discrete=True)
        self.target_num_species_reached = False

        if Config.use_representative:
            self.representative = representative

    def get_similar_modules(self, modules, n):
        if not Config.use_representative:
            raise Exception('get_similar_modules called, but use representatives is false')

        return heapq.nsmallest(n, modules, key=lambda indv: indv.distance_to(self.representative))

    def choose_representative(self, modules, all_reps):
        all_reps = list(set(all_reps))  # removing duplicated to make choosing fair
        chance = random.random()
        # If rep is none ignore chance to pick similar rep
        chance_pick_rand = 0.7
        if self.representative is None:
            chance_pick_rand = 1

        if chance < 0.5 and all_reps:
            # 50% chance to pick random from reps already in the blueprint to promote repeating structures
            self.representative = random.choice(all_reps)
        elif chance < chance_pick_rand:
            # 20% or 50% chance to pick random from pop
            new_rep = copy.deepcopy(random.choice(modules))

            for rep in all_reps:
                if new_rep == rep:
                    new_rep = rep
                    break

            self.representative = new_rep
        elif chance < 0.75:
            # 0% or 5% chance to pick a very different representative
            new_rep = copy.deepcopy(
                random.choice(heapq.nlargest(10, modules, key=lambda indv: indv.distance_to(self.representative))))

            for rep in all_reps:
                if new_rep == rep:
                    new_rep = rep
                    break

            self.representative = new_rep
        else:
            # 0% or 20% chance to pick a similar representative
            choices = self.get_similar_modules(modules, Config.closest_reps_to_consider)

            weights = [2 - (x / Config.closest_reps_to_consider) for x in
                       range(Config.closest_reps_to_consider)]  # closer reps have a higher chanecs
            self.representative = random.choices(choices, weights=weights, k=1)[0]

        return self.representative

    def get_all_mutagens(self):
        return [self.species_number]

    def set_species_upper_bound(self, num_species, generation_number):
        """used to update the species number mutagens. takes care of the species number shuffling"""
        if not self.target_num_species_reached and num_species >= Props.MODULE_TARGET_NUM_SPECIES:
            """species count starts low, and increases quickly. 
            due to low species number mutation rates, nodes would largely be stuck
            referencing species 0 for many generations before a good distribution arises
            so we force a shuffle in the early generations, to get a good distribution early"""
            self.target_num_species_reached = True
            self.species_number.mutation_chance = 0.13  # the desired stable mutation rate
            if generation_number < 3:
                """by now, the target species numbers have been reached, but the high mutation rate \
                has not had enough time to create a good distribution. so we shuffle species numbers"""
                self.species_number.set_value(random.randint(0, num_species - 1))

        self.species_number.end_range = num_species
        if self.species_number() >= num_species:
            self.species_number.set_value(random.randint(0, num_species - 1))

    def get_node_name(self):
        return "Species:" + repr(self.species_number())


class DANode(NodeGene):
    def __init__(self, id, node_type=NodeType.HIDDEN):
        """initialises da mutagens, and sets initial values"""
        super().__init__(id, node_type)

        da_submutagens = {

            "Rotate": {
                "lo": Mutagen(value_type=ValueType.WHOLE_NUMBERS, current_value=-30, start_range=-180,
                              end_range=-1, mutation_chance=0.2),
                "hi": Mutagen(value_type=ValueType.WHOLE_NUMBERS, current_value=30, start_range=0,
                              end_range=180, mutation_chance=0.2)},

            "Translate_Pixels": {
                "x_lo": Mutagen(value_type=ValueType.WHOLE_NUMBERS, current_value=-4, start_range=-15,
                                end_range=-1, mutation_chance=0.2),
                "x_hi": Mutagen(value_type=ValueType.WHOLE_NUMBERS, current_value=4, start_range=0,
                                end_range=15, mutation_chance=0.2),
                "y_lo": Mutagen(value_type=ValueType.WHOLE_NUMBERS, current_value=-4, start_range=-15,
                                end_range=-1, mutation_chance=0.2),
                "y_hi": Mutagen(value_type=ValueType.WHOLE_NUMBERS, current_value=4, start_range=0,
                                end_range=15, mutation_chance=0.2)},

            "Scale": {
                "x_lo": Mutagen(value_type=ValueType.CONTINUOUS, current_value=0.75, start_range=0.25,
                                end_range=0.99, mutation_chance=0.3),
                "x_hi": Mutagen(value_type=ValueType.CONTINUOUS, current_value=1.25, start_range=1.0,
                                end_range=2.0, mutation_chance=0.3),
                "y_lo": Mutagen(value_type=ValueType.CONTINUOUS, current_value=0.75, start_range=0.25,
                                end_range=0.99, mutation_chance=0.3),
                "y_hi": Mutagen(value_type=ValueType.CONTINUOUS, current_value=1.25, start_range=1.0,
                                end_range=2.0, mutation_chance=0.3)},

            "Pad_Pixels": {
                "lo": Mutagen(value_type=ValueType.WHOLE_NUMBERS, current_value=1, start_range=0,
                              end_range=3, mutation_chance=0.2),
                "hi": Mutagen(value_type=ValueType.WHOLE_NUMBERS, current_value=4, start_range=4,
                              end_range=6, mutation_chance=0.2),
                "s_i": Mutagen(True, False, discreet_value=False, mutation_chance=0.2)},

            "Crop_Pixels": {
                "lo": Mutagen(value_type=ValueType.WHOLE_NUMBERS, current_value=1, start_range=0,
                              end_range=3, mutation_chance=0.2),
                "hi": Mutagen(value_type=ValueType.WHOLE_NUMBERS, current_value=4, start_range=4,
                              end_range=6, mutation_chance=0.2),
                "s_i": Mutagen(True, False, discreet_value=False, mutation_chance=0.2)},

            "Custom_Canny_Edges": {
                "min_val": Mutagen(value_type=ValueType.WHOLE_NUMBERS, current_value=100, start_range=50,
                                   end_range=149, mutation_chance=0.2),
                "max_val": Mutagen(value_type=ValueType.WHOLE_NUMBERS, current_value=200, start_range=150,
                                   end_range=250, mutation_chance=0.2)},

            "Additive_Gaussian_Noise": {
                "lo": Mutagen(value_type=ValueType.CONTINUOUS, current_value=0.05, start_range=0.0,
                              end_range=0.09, mutation_chance=0.3),
                "hi": Mutagen(value_type=ValueType.CONTINUOUS, current_value=0.125, start_range=0.10,
                              end_range=0.30, mutation_chance=0.3),
                "percent": Mutagen(value_type=ValueType.CONTINUOUS, current_value=0.6, start_range=0.2,
                                   end_range=0.8, mutation_chance=0.3)
            },

            "Coarse_Dropout": {
                "d_lo": Mutagen(value_type=ValueType.CONTINUOUS, current_value=0.03, start_range=0.0,
                                end_range=0.09, mutation_chance=0.3),
                "d_hi": Mutagen(value_type=ValueType.CONTINUOUS, current_value=0.15, start_range=0.1,
                                end_range=0.3, mutation_chance=0.3),
                "s_lo": Mutagen(value_type=ValueType.CONTINUOUS, current_value=0.025, start_range=0.0,
                                end_range=0.09, mutation_chance=0.3),
                "s_hi": Mutagen(value_type=ValueType.CONTINUOUS, current_value=0.3, start_range=0.1,
                                end_range=1.0, mutation_chance=0.3),
                "percent": Mutagen(value_type=ValueType.CONTINUOUS, current_value=0.6, start_range=0.2,
                                   end_range=0.8, mutation_chance=0.3)
            },

        }

        if Config.colour_augmentations:

            da_submutagens["HSV"] = {
                "channel": Mutagen(0, 1, 2, discreet_value=0, mutation_chance=0.1),
                "lo": Mutagen(value_type=ValueType.WHOLE_NUMBERS, current_value=20, start_range=0,
                              end_range=29, mutation_chance=0.2),
                "hi": Mutagen(value_type=ValueType.WHOLE_NUMBERS, current_value=50, start_range=30,
                              end_range=60, mutation_chance=0.2)
            }
            da_submutagens["Grayscale"] = {
                "alpha_lo": Mutagen(value_type=ValueType.CONTINUOUS, current_value=0.35, start_range=0.0,
                                    end_range=0.49, mutation_chance=0.3),
                "alpha_hi": Mutagen(value_type=ValueType.CONTINUOUS, current_value=0.75, start_range=0.5,
                                    end_range=1.0, mutation_chance=0.3)}

            self.da = Mutagen("Flip_lr", "Rotate", "Translate_Pixels", "Scale", "Pad_Pixels", "Crop_Pixels",
                              "Grayscale", "Custom_Canny_Edges", "Additive_Gaussian_Noise", "Coarse_Dropout",
                              "HSV", "No_Operation", name="da type", sub_mutagens=da_submutagens,
                              discreet_value=random.choice(list(da_submutagens.keys())), mutation_chance=0.25)
        else:

            self.da = Mutagen("Flip_lr", "Rotate", "Translate_Pixels", "Scale", "Pad_Pixels", "Crop_Pixels",
                              "Custom_Canny_Edges", "Additive_Gaussian_Noise", "Coarse_Dropout",
                              "No_Operation", name="da type", sub_mutagens=da_submutagens,
                              discreet_value=random.choice(list(da_submutagens.keys())), mutation_chance=0.25)

        self.enabled = Mutagen(True, False, discreet_value=True, name="da enabled")

    def get_all_mutagens(self):
        return [self.da, self.enabled]

    def get_node_name(self):
        return repr(self.da()) + "\n" + self.get_node_parameters()

    def get_node_parameters(self):
        """used for plotting da genomes"""
        parameters = []
        if self.da.get_sub_values() is not None:
            for key, value in self.da.get_sub_values().items():

                if value is None:
                    raise Exception("none value in mutagen")

                v = repr(value())

                parameters.append((key, v))

        return repr(parameters)
