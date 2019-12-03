from __future__ import annotations

from typing import TYPE_CHECKING

from src2.Configuration import config
from src2.Evolution import Run
from src2.Phenotype.NeuralNetwork.NeuralNetwork import Network
from src2.Phenotype.NeuralNetwork.Evaluator.DataLoader import get_data_shape
from src2.Phenotype.NeuralNetwork.Evaluator.Evaluator import evaluate


if TYPE_CHECKING:
    from src2.Phenotype import NeuralNetwork
    from src2.Evolution.Run import Run as RunClass


def fully_train_nn(model: NeuralNetwork, num_epochs):
    accuracy = evaluate(model, num_epochs=num_epochs, fully_training= True)
    print("model scored " , accuracy, " on fully training")

def fully_train_best_evolved_networks(run_name, n=1):
    run: RunClass = Run.get_run(run_name)
    best_blueprints = run.get_most_accurate_blueprints(n)
    in_size = get_data_shape()
    import src2.main.Singleton as S

    for blueprint, gen_number in best_blueprints:
        S.instance = run.generations[gen_number]
        modules = run.get_modules_for_blueprint(blueprint)
        print("bp:", blueprint, "\nmodules:", modules, "\nsample map:", blueprint.best_module_sample_map,
              "\nspecies used:", list(set([x.species_id for x in blueprint.nodes.values()])))

        device = config.get_device()
        model: Network = Network(blueprint, in_size).to(device)
        fully_train_nn(model, 50)