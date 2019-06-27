from src.EvolutionEnvironment.Generation import Generation

"""
Evolution Environment is static as there should only ever be one
Acts as the collection of current generation
"""

numGenerations = 5


def main():
    current_generation = Generation()

    for i in range(numGenerations):
        current_generation.evaluate()
        current_generation.step()



if __name__ == '__main__':
    main()
