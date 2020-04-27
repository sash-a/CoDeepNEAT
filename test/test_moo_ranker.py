import random

from configuration.config_loader import get_cli_args, load_simple_config
from src.genotype.cdn.genomes.blueprint_genome import BlueprintGenome
from src.genotype.neat.operators.population_rankers.two_objective_rank import TwoObjectiveRank
from src.utils.wandb_utils import resume_evo_run, new_evo_run

import matplotlib.pyplot as plt

from src.utils.main_common import init_generation

args = get_cli_args()
load_simple_config(args.config, resume_evo_run, new_evo_run, args.ngpus)
generation = init_generation()


num_individuals = 100000
indv_to_no_map = {}
individuals = [BlueprintGenome([],[]) for i in range(num_individuals)]
for indv_no in range(len(individuals)):
    indv = individuals[indv_no]
    indv.fitness_values = [random.uniform(0,100), random.uniform(0,100)]
    indv_to_no_map[indv] = indv_no

moo_ranker = TwoObjectiveRank()
fronts = moo_ranker.cdn_rank(individuals)
num_fronts = len(fronts)

print("num fronts: ", num_fronts)

indv_to_front_map = {}
for front_no in range(num_fronts):
    for indv in fronts[front_no]:
        indv_to_front_map[indv] = front_no

xs = [indv.fitness_values[0] for indv in individuals]
ys = [indv.fitness_values[1] for indv in individuals]
ranks = [indv.rank for indv in individuals]


front_colouring = lambda indv: (indv_to_front_map[indv] % 2, (indv_to_front_map[indv] % 3)/2.0, (indv_to_front_map[indv] % 4)/3.0)

# colours = [TwoObjectiveRank.get_rank_colour(indv, num_individuals) for indv in individuals]
colours = [front_colouring(indv) for indv in individuals]


for front_no in range(num_fronts):
    label = "front:" + str(front_no)
    xxs = []
    yys = []

    for indv in fronts[front_no]:
        i = indv_to_no_map[indv]
        colour = colours[i]
        xxs.append(xs[i])
        yys.append(ys[i])

    plt.scatter(xxs,yys,color=colour)
    # plt.scatter(xxs,yys,color=colour, label=label)


plt.legend()
plt.show()