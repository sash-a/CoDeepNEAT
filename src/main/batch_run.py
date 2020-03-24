from __future__ import annotations

import atexit
import os
import sys

# For importing project files
dir_path = os.path.dirname(os.path.realpath(__file__))
dir_path_1 = os.path.split(os.path.split(dir_path)[0])[0]
sys.path.append(dir_path_1)
sys.path.append(os.path.join(dir_path_1, 'src'))
sys.path.append(os.path.join(dir_path_1, 'test'))
sys.path.append(os.path.join(dir_path_1, 'runs'))
sys.path.append(os.path.join(dir_path_1, 'configuration'))

from src.utils.main_common import _force_cuda_device_init, evolve
from configuration import internal_config, config_loader, config
from src.phenotype.neural_network.evaluator.fully_train import fully_train


def main():
    config_loader.load_batch_config()

    _force_cuda_device_init()

    if config.fully_train:
        fully_train(config.run_name)
    else:
        evolve()


if __name__ == '__main__':
    atexit.register(internal_config.on_exit)
    main()
