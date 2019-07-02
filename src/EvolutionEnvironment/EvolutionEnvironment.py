from src.EvolutionEnvironment.Generation import Generation
import torch
from src.Analysis import RuntimeAnalysis

"""
Evolution Environment is static as there should only ever be one
Acts as the collection of current generation
"""

numGenerations = 500


def main():
    current_generation = Generation()
    RuntimeAnalysis.configure(log_file_name="test")

    for i in range(numGenerations):
        print('Running gen', i)
        current_generation.evaluate(i, print_graphs=False)
        current_generation.step()


if __name__ == '__main__':
    main()
