from src.EvolutionEnvironment.Generation import Generation

"""
Evolution Environment is static as there should only ever be one
Acts as the collection of current generation
"""

numGenerations = 1

currentGeneration = None
previousGeneration = None


def main():
    current_generation = Generation(True)

    for i in range(numGenerations):
        current_generation.evaluate()


if __name__ == '__main__':
    main()
