from src.EvolutionEnvironment.Generation import Generation
from src.Analysis import RuntimeAnalysis
import time

"""
Evolution Environment is static as there should only ever be one
Acts as the driver of current generation
"""

num_generations = 500
protect_parsing_from_errors = True
current_generation = None


def main():
    global current_generation

    current_generation = Generation()
    RuntimeAnalysis.configure(log_file_name="test")
    start_time = time.time()

    for i in range(num_generations):
        print('Running gen', i)
        gen_start_time = time.time()
        current_generation.evaluate(i)
        current_generation.step()
        print('completed gen', i, "in", (time.time() - gen_start_time), "elapsed time:", (time.time() - start_time),"\n\n")


if __name__ == '__main__':
    main()
