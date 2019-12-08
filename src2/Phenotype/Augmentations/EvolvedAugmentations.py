from __future__ import annotations
from typing import Dict, Any, Union, TYPE_CHECKING

import imgaug.augmenters as iaa
import random
from src2.Configuration import config

from src2.Genotype.Mutagen.IntegerVariable import IntegerVariable
from src2.Genotype.Mutagen.ContinuousVariable import ContinuousVariable
from src2.Genotype.Mutagen.Option import Option

import src2.Phenotype.Augmentations.DADefinitions as DAD

# augmentations = [
#     iaa.Fliplr(1),
#     iaa.Affine(rotate=(-30, 30)),
#     iaa.Affine(translate_px={"x": (-4, 4), "y": (-4, 4)}),
#     iaa.Affine(scale={"x": (0.75, 1.25), "y": (0.75, 1.25)}),
#     iaa.Pad(px=(1, 4), keep_size=True, sample_independently=False),
#     iaa.Crop(px=(1, 4), keep_size=True, sample_independently=False),
#     iaa.CoarseDropout((0.05, 0.2), size_percent=(0.025, 0.5), per_channel=0.6),
#     iaa.WithColorspace(to_colorspace="HSV", from_colorspace="RGB", children=iaa.WithChannels(0, iaa.Add((20, 50)))),
#     iaa.Grayscale(alpha=(0.35, 0.75)),
#     iaa.Noop()
# ]

# IntegerVariable("name", current_value, starting_range, ending_range, mutation_chance)
# ContinuousVariable("name", current_value, starting_range, ending_range, mutation chance)
# Option("name", options*, current_value, mutation_chance)

DA_SubMutagens = {

    "Rotate": {
        "lo": IntegerVariable("lo", -30, -180, 1, 0.2),
        "hi": IntegerVariable("hi", 30, 0, 180, 0.2)},

    "Translate_Pixels": {
        "x_lo": IntegerVariable("x_lo", -4, -15, -1, 0.2),
        "x_hi": IntegerVariable("x_hi", 4, 0, 15, 0.2),
        "y_lo": IntegerVariable("y_lo", -4, -15, -1, 0.2),
        "y_hi": IntegerVariable("y_hi", 4, 0, 15, 0.2)},

    "Scale": {
        "x_lo": ContinuousVariable("x_lo", 0.75, 0.25, 0.99, 0.3),
        "x_hi": ContinuousVariable("x_hi", 1.25, 1.0, 2.0, 0.3),
        "y_lo": ContinuousVariable("y_lo", 0.75, 0.25, 0.99, 0.3),
        "y_hi": ContinuousVariable("y_hi", 1.25, 1.0, 2.0, 0.3)},

    "Pad_Pixels": {
        "lo": IntegerVariable("lo", 1, 0, 3, 0.2),
        "hi": IntegerVariable("hi", 4, 4, 6, 0.2),
        "s_i": Option("s_i", True, False, current_value=False, mutation_chance=0.2)},

    "Crop_Pixels": {
        "lo": IntegerVariable("lo", 1, 0, 3, 0.2),
        "hi": IntegerVariable("hi", 4, 4, 6, 0.2),
        "s_i": Option("s_i", True, False, current_value=False, mutation_chance=0.2)},


    "Coarse_Dropout": {
        "d_lo": ContinuousVariable("d_lo", 0.03, 0.0, 0.09, 0.3),
        "d_hi": ContinuousVariable("d_hi", 0.15, 0.1, 0.3, 0.3),
        "s_lo": ContinuousVariable("s_lo", 0.025, 0.0, 0.09, 0.3),
        "s_hi": ContinuousVariable("s_hi", 0.3, 0.1, 1.0, 0.3),
        "percent": ContinuousVariable("percent", 0.6, 0.2, 0.8, 0.3)
    }

}

# Separated Photometric (Colour) Augmentations from geometric (non-colour) ones
if config.use_colour_augmentations:

    DA_SubMutagens["HSV"] = {
        "channel": Option("channel", 0, 1, 2, current_value=0, mutation_chance=0.1),
        "lo": IntegerVariable("lo", 20, 0, 29, 0.2),
        "hi": IntegerVariable("hi", 50, 30, 60, 0.2)
    }

    DA_SubMutagens["Grayscale"] = {
        "alpha_lo": ContinuousVariable("alpha_lo", 0.35, 0.0, 0.49, 0.3),
        "alpha_hi": ContinuousVariable("alpha_hi", 0.75, 0.5, 1.0, 0.3)}

    DA_Mutagens = Option("DA Type", "Flip_lr", "Rotate", "Translate_Pixels", "Scale", "Pad_Pixels", "Crop_Pixels",
                         "Grayscale", "Coarse_Dropout", "HSV", "No_Operation",
                         current_value=random.choice(list(DA_SubMutagens.keys())),
                         submutagens=DA_SubMutagens, mutation_chance=0.25)
else:

    Option("DA Type", "Flip_lr", "Rotate", "Translate_Pixels", "Scale", "Pad_Pixels", "Crop_Pixels",
           "Coarse_Dropout", "No_Operation", current_value=random.choice(list(DA_SubMutagens.keys())),
           submutagens=DA_SubMutagens, mutation_chance=0.25)

