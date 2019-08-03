from CoDeepNEAT.CDNGenomes import BlueprintGenome
from src.NEAT.Connection import Connection

from Test.TestingGenomes import moo_pop_members, initial_mutations, nodes, connections
from src.NEAT.Population import Population
from src.NEAT.Species import Species


def test_init():
    pop = Population(moo_pop_members, initial_mutations, 10, 0, 0, 0)
    assert pop.curr_innov == 2
    assert pop.max_node_id == 2


def test_speciate():
    pop = Population(moo_pop_members, initial_mutations, 10, 0, 0, 0)
    pop.individuals.append(BlueprintGenome(connections, nodes))
    pop.individuals.append(BlueprintGenome(connections[:-1], nodes))

    fake_species_rep = BlueprintGenome(connections, nodes)
    for i in range(10):
        fake_species_rep.add_connection(Connection(nodes[0], nodes[1], innovation=i + 10))

    fake_species = Species(fake_species_rep)
    pop.speciation_thresh = 0.35
    pop.species.append(fake_species)

    pop.speciate(first_gen=False)
    assert len(pop.species) == 2

    # Making sure that the empty species was deleted
    reps = [spc.representative for spc in pop.species]
    assert fake_species.representative not in reps


# Can this be tested?
def test_dynamic_speciation():
    pass


def test_adjust_fitness():
    pass


def test_save_elite():
    pass


# Can this be tested?
def test_step():
    pass
