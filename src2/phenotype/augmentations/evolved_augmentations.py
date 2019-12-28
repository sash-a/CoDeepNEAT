from __future__ import annotations

import random

from src2.configuration import config
from src2.genotype.mutagen.continuous_variable import ContinuousVariable
from src2.genotype.mutagen.integer_variable import IntegerVariable
from src2.genotype.mutagen.option import Option


def get_da_submutagens():
    submutagens = {
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
    if config.use_colour_augmentations:
        submutagens["HSV"] = {
            "channel": Option("channel", 0, 1, 2, current_value=0, mutation_chance=0.1),
            "lo": IntegerVariable("lo", 20, 0, 29, 0.2),
            "hi": IntegerVariable("hi", 50, 30, 60, 0.2)
        }

        submutagens["Grayscale"] = {
            "alpha_lo": ContinuousVariable("alpha_lo", 0.35, 0.0, 0.49, 0.3),
            "alpha_hi": ContinuousVariable("alpha_hi", 0.75, 0.5, 1.0, 0.3)}

    return submutagens
