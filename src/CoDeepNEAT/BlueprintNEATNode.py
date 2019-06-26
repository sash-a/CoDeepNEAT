from src.NEAT.NEATNode import NEATNode

class BlueprintNEATNode(NEATNode):

    def __init__(self, id, x):
        super(BlueprintNEATNode, self).__init__(id,x)
        self.species_number = 0

    pass