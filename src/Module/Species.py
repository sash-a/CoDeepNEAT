from random import randint

class Species:

    """
    species contains a list of module input nodes
    """

    moduleCollection = None
    speciesNumber = -1

    def __init__(self):
        pass

    def sampleModule(self):
        index = randint(0,len(self.moduleCollection))
        return self.moduleCollection[index], index

