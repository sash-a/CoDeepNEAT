from src.EvolutionEnvironment.Generation import Generation
import torch

"""
Evolution Environment is static as there should only ever be one
Acts as the collection of current generation
"""

numGenerations = 500


def main():
    current_generation = Generation()

    for i in range(numGenerations):
        print('\nRunning gen', i)
        current_generation.evaluate(i, print_graphs=False, device=torch.device('cpu'))
        current_generation.step()


if __name__ == '__main__':
    main()
