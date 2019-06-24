from src.NEAT.Connection import Connection
from src.NEAT.Node import Node, NodeType
from src.NEAT.Genotype import Genome
from src.NEAT.Species import Species


class Population:
    def __init__(self, nodes: dict, connections: set, population: list):
        """
        :param nodes: dictionary mapping node IDs to nodes
        :param connections: set of all connections
        :param population: all individuals
        """
        self.nodes = nodes
        self.connections = connections
        self.population = population
        self.species = []

        self.speciate()

    def speciate(self):
        """
        Place all individuals in their first compatible species if one exists
        Otherwise create new species with current individual as its representative

        :return: list of species
        """
        # Remove all current members from the species
        for spc in self.species:
            spc.clear()

        # Placing individuals in their correct species
        for individual in self.population:
            found_species = False
            for spc in self.species:
                if spc.is_compatible(individual):
                    spc.add_member(individual)
                    found_species = True
                    break

            if not found_species:
                self.species.append(Species(individual))

        # Remove all empty species
        species = [spc for spc in self.species if spc.members]

        return species

    def run(self):
        print('THIS IS NOT PROPERLY IMPLEMENTED')
        for indv in self.population:
            # get fitness

            indv.crossover(None)  # needs to crossover with something
            indv.mutate()

            self.speciate()


def main():
    nodes = [Node(0, NodeType.INPUT), Node(1, NodeType.HIDDEN), Node(2, NodeType.OUTPUT)]
    connections = \
        [
            Connection(nodes[0], nodes[1], innovation=0),
            Connection(nodes[1], nodes[2], innovation=1),
        ]

    indv1 = Genome(connections, nodes)
    indv2 = Genome([Connection(nodes[1], nodes[2], innovation=1)], nodes)

    spc = Species(indv1)
    print(spc.is_compatible(indv2))


if __name__ == '__main__':
    main()
