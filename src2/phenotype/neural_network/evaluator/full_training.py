from __future__ import annotations

from typing import TYPE_CHECKING

from src2 import run
from src2.configuration import config
from src2.phenotype.neural_network.evaluator.data_loader import get_data_shape
from src2.phenotype.neural_network.evaluator.evaluator import evaluate
from src2.phenotype.neural_network.neural_network import Network

if TYPE_CHECKING:
    from src2.phenotype import neural_network
    from src2.run import Run as RunClass


def fully_train_nn(model: neural_network, num_epochs):
    accuracy = evaluate(model, num_epochs=num_epochs, fully_training=True, augmentation_transform=model.blueprint.da_scheme.to_phenotype())
    print("model scored ", accuracy, " on fully training")


def fully_train_best_evolved_networks(run_name, n=1, num_epochs=100):
    run: RunClass = run.get_run(run_name)
    best_blueprints = run.get_most_accurate_blueprints(n)
    in_size = get_data_shape()
    import src2.main.singleton as S

    for blueprint, gen_number in best_blueprints:
        S.instance = run.generations[gen_number]
        modules = run.get_modules_for_blueprint(blueprint)
        print("bp:", blueprint, "\nmodules:", modules, "\nsample map:", blueprint.best_module_sample_map,
              "\nspecies used:", list(set([x.species_id for x in blueprint.nodes.values()])))
        device = config.get_device()
        model: Network = Network(blueprint, in_size, sample_map=blueprint.best_module_sample_map).to(device)
        model_size = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("training model which scored:", blueprint.accuracy, " in evolution for ", num_epochs, "epochs, with",
              model_size, " params")
        blueprint.visualize()
        model.visualize()
        fully_train_nn(model, num_epochs)
