from __future__ import annotations

import random

from configuration import config
from src.genotype.mutagen.continuous_variable import ContinuousVariable
from src.genotype.mutagen.integer_variable import IntegerVariable
from src.genotype.mutagen.option import Option


def get_da_submutagens():
    submutagens = {
        "Rotate": {
            "lo": IntegerVariable("lo", random.randint(-180, 1), -180, 1, 0.2),
            "hi": IntegerVariable("hi", random.randint(0, 180), 0, 180, 0.2)},

        "Translate_Pixels": {
            "x_lo": IntegerVariable("x_lo", random.randint(-15, -1), -15, -1, 0.2),
            "x_hi": IntegerVariable("x_hi", random.randint(0, 15), 0, 15, 0.2),
            "y_lo": IntegerVariable("y_lo", random.randint(-15, -1), -15, -1, 0.2),
            "y_hi": IntegerVariable("y_hi", random.randint(0, 15), 0, 15, 0.2)},

        "Scale": {
            "x_lo": ContinuousVariable("x_lo", random.uniform(0.25, 0.99), 0.25, 0.99, 0.3),
            "x_hi": ContinuousVariable("x_hi", random.uniform(1.0, 2.0), 1.0, 2.0, 0.3),
            "y_lo": ContinuousVariable("y_lo", random.uniform(0.25, 0.99), 0.25, 0.99, 0.3),
            "y_hi": ContinuousVariable("y_hi", random.uniform(1.0, 2.0), 1.0, 2.0, 0.3)},

        "Pad_Pixels": {
            "lo": IntegerVariable("lo", random.randint(0, 3), 0, 3, 0.2),
            "hi": IntegerVariable("hi", random.randint(4, 6), 4, 6, 0.2),
            "s_i": Option("s_i", True, False, current_value=random.choice([True, False]), mutation_chance=0.2)},

        "Crop_Pixels": {
            "lo": IntegerVariable("lo", random.randint(0, 3), 0, 3, 0.2),
            "hi": IntegerVariable("hi", random.randint(4, 6), 4, 6, 0.2),
            "s_i": Option("s_i", True, False, current_value=random.choice([True, False]), mutation_chance=0.2)},

        "Coarse_Dropout": {
            "d_lo": ContinuousVariable("d_lo", random.uniform(0.0, 0.09), 0.0, 0.09, 0.3),
            "d_hi": ContinuousVariable("d_hi", random.uniform(0.1, 0.3), 0.1, 0.3, 0.3),
            "s_lo": ContinuousVariable("s_lo", random.uniform(0.0, 0.09), 0.0, 0.09, 0.3),
            "s_hi": ContinuousVariable("s_hi", random.uniform(0.1, 1.0), 0.1, 1.0, 0.3),
            "percent": ContinuousVariable("percent", random.uniform(0.2, 0.8), 0.2, 0.8, 0.3)
        }
    }
    if config.use_colour_augmentations:
        submutagens["HSV"] = {
            "channel": Option("channel", 0, 1, 2, current_value=random.choice([0, 1, 2]), mutation_chance=0.1),
            "lo": IntegerVariable("lo", random.randint(0, 29), 0, 29, 0.2),
            "hi": IntegerVariable("hi", random.randint(30, 60), 30, 60, 0.2)
        }

        submutagens["Grayscale"] = {
            "alpha_lo": ContinuousVariable("alpha_lo", random.uniform(0.0, 0.49), 0.0, 0.49, 0.3),
            "alpha_hi": ContinuousVariable("alpha_hi", random.uniform(0.5, 1.0), 0.5, 1.0, 0.3)}

    return submutagens


def get_legacy_da_submutagens():
    submutagens = {
        "Scale": {
            "x_lo": ContinuousVariable("x_lo", random.uniform(1.0, 1.14), 1.0, 1.14, 0.3),
            "x_hi": ContinuousVariable("x_hi", random.uniform(1.15, 1.3), 1.15, 1.3, 0.3),

            "y_lo": ContinuousVariable("y_lo", random.uniform(1.0, 1.14), 1.0, 1.15, 0.3),
            "y_hi": ContinuousVariable("y_hi", random.uniform(1.15, 1.3), 1.15, 1.3, 0.3)},

        "Crop_Pixels": {
            "lo": IntegerVariable("lo", random.randint(0, 3), 0, 3, 0.2),
            "hi": IntegerVariable("hi", random.randint(4, 6), 4, 6, 0.2),
            "s_i": Option("s_i",  False, current_value=random.choice([True, False]), mutation_chance=0)}

    }
    if config.use_colour_augmentations:
        submutagens["HSV"] = {
            "channel": Option("channel", 0, 1, 2, current_value=random.choice([0, 1, 2]), mutation_chance=0),
            "lo": IntegerVariable("lo", random.randint(0, 27), 0, 27, 0.2),
            "hi": IntegerVariable("hi", random.randint(27, 45), 27, 45, 0.2)
        }

    return submutagens
