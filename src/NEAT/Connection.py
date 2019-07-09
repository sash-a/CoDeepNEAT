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
        return str(self.from_node) + ' -> ' + str(self.to_node) + ' innov: ' + str(self.innovation)

    def __eq__(self, other):
        return self.from_node == other.from_node and self.to_node == other.to_node

    def __hash__(self):
        return self.innovation

    def get_all_mutagens(self):
        return [self.enabled]

    def mutate(self):
        for mutagen in self.get_all_mutagens():
            mutagen.mutate()

    def isvalid(self):
        return self.from_node.x <= self.to_node.x and self.from_node != self.to_node
