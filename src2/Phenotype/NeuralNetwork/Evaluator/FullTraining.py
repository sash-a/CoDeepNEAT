from __future__ import annotations

from typing import TYPE_CHECKING

from src2.Configuration import config
from src2 import Run
from src2.Phenotype.NeuralNetwork.Evaluator.DataLoader import get_data_shape
from src2.Phenotype.NeuralNetwork.Evaluator.Evaluator import evaluate
from src2.Phenotype.NeuralNetwork.NeuralNetwork import Network

if TYPE_CHECKING:
    from src2.Phenotype import NeuralNetwork
    from src2.Run import Run as RunClass


def fully_train_nn(model: NeuralNetwork, num_epochs):
    accuracy = evaluate(model, num_epochs=num_epochs, fully_training=True)
    print("model scored ", accuracy, " on fully training")


def fully_train_best_evolved_networks(run_name, n=1, num_epochs=100):
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
        model: Network = Network(blueprint, in_size, prescribed_sample_map=blueprint.best_module_sample_map).to(device)
        model_size = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("training model which scored:", blueprint.accuracy, " in evolution for ", num_epochs, "epochs, with",
              model_size, " params")
        blueprint.visualize()
        model.visualize()
        fully_train_nn(model, num_epochs)
