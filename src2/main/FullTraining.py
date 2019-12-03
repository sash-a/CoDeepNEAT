from __future__ import annotations

from typing import TYPE_CHECKING

from src2.Evolution import Run

if TYPE_CHECKING:
    from src2.Phenotype import NeuralNetwork
    from src2.Evolution.Run import Run as RunClass


def fully_train_nn(model: NeuralNetwork, num_epochs):
    pass


def fully_train_best_evolved_networks(run_name, n=1):
    run: RunClass = Run.get_run(run_name)
    best_blueprints = run.get_most_accurate_blueprints(n)
    for blueprint, gen in best_blueprints:
        modules = run.get_modules_for_blueprint(blueprint)
        print("bp:", blueprint, "modules:", modules)
