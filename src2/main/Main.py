from __future__ import annotations
import argparse
from typing import TYPE_CHECKING

from src2.Configuration import config
from src2.main.Generation import Generation

if TYPE_CHECKING:
    from src2.Phenotype import NeuralNetwork


def main():
    parse_config()

    generation = Generation()
    for i in range(config.n_generations):
        print('Started generation:', i)
        evolve_generation(generation)


def parse_config():
    parser = argparse.ArgumentParser(description='CoDeepNEAT')
    parser.add_argument('-c', '--configs', nargs='+', type=str,
                        help='Path to all config files that will be used. (Earlier configs are given preference)',
                        required=False)
    args = parser.parse_args()
    # Reading configs in reverse order so that initial config files overwrite others
    if args.configs is None:
        return

    for cfg_file in reversed(args.configs):
        config.read(cfg_file)


def evolve_generation(generation: Generation):
    generation.step()


def fully_train_nn(model: NeuralNetwork):
    pass


if __name__ == '__main__':
    main()
