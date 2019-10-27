"""
    the generation class is a container for the 3 cdn populations.
    It is also responsible for stepping the evolutionary cycle.
"""


class Generation:

    def __init__(self):
        self.module_population, self.blueprint_population, self.da_population = None, None, None

    def evaluate_blueprints(self):
        """
            evaluates all blueprints multiple times.
            passes evaluation scores back to individuals
        """
        pass

    def initialise_populations(self):
        """starts off the populations of a new evolutionary run"""
        pass

    def step(self):
        """
            Runs CDN for one generation
            calls the evaluation of all individuals
            prepares population objects for the next step
        """
