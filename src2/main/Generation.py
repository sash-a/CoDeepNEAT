"""
    the generation class is a container for the 3 cdn populations.
    It is also responsible for stepping the evolutionary cycle.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import wandb

import src2.main.Singleton as Singleton
from src2.Configuration import config
from src2.Genotype.CDN.Genomes.BlueprintGenome import BlueprintGenome
from src2.Genotype.CDN.Genomes.ModuleGenome import ModuleGenome
from src2.Genotype.CDN.Nodes.BlueprintNode import BlueprintNode
from src2.Genotype.CDN.Nodes.ModuleNode import ModuleNode
from src2.Genotype.CDN.Operators.Mutators.BlueprintGenomeMutator import BlueprintGenomeMutator
from src2.Genotype.CDN.Operators.Mutators.ModuleGenomeMutator import ModuleGenomeMutator
from src2.Genotype.CDN.PopulationInitializer import create_population, create_mr
from src2.Genotype.NEAT.Operators.PopulationRankers.SingleObjectiveRank import SingleObjectiveRank
from src2.Genotype.NEAT.Operators.RepresentativeSelectors.BestRepSelector import BestRepSelector
from src2.Genotype.NEAT.Operators.RepresentativeSelectors.MostSimilarRepSelector import MostSimilarRepSelector
from src2.Genotype.NEAT.Operators.RepresentativeSelectors.RandomRepSelector import RandomRepSelector
from src2.Genotype.NEAT.Operators.ParentSelectors.RouletteSelector import RouletteSelector
from src2.Genotype.NEAT.Operators.ParentSelectors.TournamentSelector import TournamentSelector
from src2.Genotype.NEAT.Operators.ParentSelectors.UniformSelector import UniformSelector
from src2.Genotype.NEAT.Operators.Speciators.MostSimilarSpeciator import MostSimilarSpeciator
from src2.Genotype.NEAT.Operators.Speciators.NEATSpeciator import NEATSpeciator
from src2.Genotype.NEAT.Population import Population
from src2.Genotype.NEAT.Species import Species
from src2.Phenotype.NeuralNetwork.Evaluator.DataLoader import get_data_shape
from src2.Phenotype.NeuralNetwork.PhenotypeEvaluator import evaluate_blueprint


class Generation:
    def __init__(self):

        Singleton.instance = self

        if not config.multiobjective:
            Population.ranker = SingleObjectiveRank()
        else:
            # TODO multiobjective rank
            raise NotImplemented('Multi-objectivity is not yet implemented')

        if config.parent_selector.lower() == "uniform":
            Species.selector = UniformSelector()
        elif config.parent_selector.lower() == "roulette":
            Species.selector = RouletteSelector()
        elif config.parent_selector.lower() == "tournament":
            Species.selector = TournamentSelector(5)
        else:
            raise Exception("unrecognised parent selector in config: " + str(config.parent_selector).lower() +
                            " expected either: uniform | roulette | tournament")

        if config.representative_selector.lower() == "centroid":
            Species.representative_selector = MostSimilarRepSelector()
        elif config.representative_selector.lower() == "random":
            Species.representative_selector = RandomRepSelector()
        elif config.representative_selector.lower() == "best":
            Species.representative_selector = BestRepSelector()
        else:
            raise Exception("unrecognised representative selector in config: " + config.representative_selector.lower()
                            + " expected centroid | random | best")

        self.module_population: Optional[Population] = None
        self.blueprint_population: Optional[Population] = None
        self.da_population: Optional[Population] = None

        self.initialise_populations()

        if config.use_wandb:
            tags = []  # TODO: add in module retention, speciation, DA

            if not tags:
                tags = ['base']

            wandb.init(name=config.run_name, project='cdn_test', tags=tags, dir='../../results')
            wandb.config.dataset = config.dataset
            wandb.config.evolution_epochs = config.epochs_in_evolution
            wandb.config.generations = config.n_generations

    def step(self):
        """
            Runs CDN for one generation. Calls the evaluation of all individuals. Prepares population objects for the
            next step.
        """
        self.evaluate_blueprints()  # may be parallel

        if config.use_wandb:
            self.wandb_report()

        if config.plot_best_genotypes:
            self.blueprint_population.get_most_accurate().visualize()
        print("maxacc:", self.blueprint_population.get_most_accurate().fitness_values)

        self.module_population.step()
        self.blueprint_population.step()

        print('Step ended')
        print('Module species:', [len(spc.members) for spc in self.module_population.species])

    def evaluate_blueprints(self):
        """Evaluates all blueprints"""
        # Multiplying the blueprints so that each blueprint is evaluated config.n_evaluations_per_bp times
        blueprints = list(self.blueprint_population) * config.n_evaluations_per_bp
        in_size = get_data_shape()

        if config.n_gpus > 1:
            with ThreadPoolExecutor(max_workers=config.n_gpus, thread_name_prefix='thread') as ex:
                results = ex.map(lambda x: evaluate_blueprint(*x), list(zip(blueprints, [in_size] * len(blueprints))))
                # results = ex.map(evaluate_blueprint, blueprints, input_size)
                for result in results:
                    r = result

            # reset_thread_name()
        else:
            for bp in blueprints:
                # print('running in series')
                evaluate_blueprint(bp, in_size)

    def initialise_populations(self):
        """Starts off the populations of a new evolutionary run"""

        if config.module_speciation.lower() == "similar":
            module_speciator = MostSimilarSpeciator(config.species_distance_thresh_mod_base, config.n_module_species,
                                                    ModuleGenomeMutator())
        elif config.module_speciation.lower() == "neat":
            module_speciator = NEATSpeciator(config.species_distance_thresh_mod_base, config.n_module_species,
                                             ModuleGenomeMutator())
        else:
            raise Exception(
                "speciation method in config not recognised: " + str(config.module_speciation).lower()
                + " expected: similar | neat")

        bp_speciator = NEATSpeciator(config.species_distance_thresh_mod_base, config.n_blueprint_species,
                                     BlueprintGenomeMutator())

        self.module_population = Population(create_population(config.module_pop_size, ModuleNode, ModuleGenome),
                                            create_mr(), config.module_pop_size, module_speciator)
        self.blueprint_population = Population(create_population(config.bp_pop_size, BlueprintNode, BlueprintGenome),
                                               create_mr(), config.bp_pop_size, bp_speciator)
        # TODO DA pop

    def wandb_report(self):
        module_accs = sorted([module.fitness_values[0] for module in self.module_population])
        bp_accs = sorted([bp.fitness_values[0] for bp in self.blueprint_population])

        mod_acc_tbl = wandb.Table(['module accuracies'], data=module_accs)
        bp_acc_tbl = wandb.Table(['blueprint accuracies'], data=bp_accs)

        wandb.log({'module accuracy table': mod_acc_tbl, 'blueprint accuracy table': bp_acc_tbl,
                   'module accuracies raw': module_accs, 'blueprint accuracies raw': bp_accs,
                   'avg module accuracy': sum(module_accs) / len(module_accs),
                   'avg blueprint accuracy': sum(bp_accs) / len(bp_accs),
                   'best module accuracy': module_accs[-1], 'best blueprint accuracy': bp_accs[-1],
                   'num module species': len(self.module_population.species),
                   'species sizes': [len(spc.members) for spc in self.module_population.species]})
