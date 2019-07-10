from src.NEAT.NEATNode import NEATNode
from src.NEAT.Mutagen import Mutagen


class Connection:
    def __init__(self, from_node: NEATNode, to_node: NEATNode, connection_type=None, enabled=True, innovation=0):
        self.from_node: NEATNode = from_node
        self.to_node: NEATNode = to_node
        self.connection_type = connection_type
        self.enabled = Mutagen(True, False, discreet_value=enabled)
        self.innovation = innovation

    def validate(self):
        if self.from_node.x > self.to_node.x:
            print('Issue with to connection')

        if self.from_node.x > self.to_node.x:
            print('Issue with from connection')

    def __repr__(self):
        return str(self.from_node) + ' -> ' + str(self.to_node) + ' innov: ' + str(self.innovation) + ' (' + str(
            self.enabled.get_value()) + ')'

    def __eq__(self, other):
        return self.from_node == other.from_node and self.to_node == other.to_node

    def __hash__(self):
        return self.innovation

    def get_all_mutagens(self):
        return [self.enabled]

    def mutate(self, genome):
        return
        mutated = False
        for mutagen in self.get_all_mutagens():
            mutated = mutagen.mutate() or mutated
        if mutated:
            """if the gene mutated - and the connection was disabled - a graph validation must be performed"""
            if not self.enabled():
                # print("turned off a connection")
                if not genome.validate():
                    """the connection was turned off - and the graph is not valid"""
                    print("turning connection back on as it was off and part of an invalid genome")
                    self.enabled.set_value(True)

    def isvalid(self):
        return self.from_node.x <= self.to_node.x and self.from_node != self.to_node
