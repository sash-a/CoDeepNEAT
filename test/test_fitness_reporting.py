genome_fitnesses = {}  # id: fits
mod_fitness = {}  # fit: ids


with open('output.txt', 'r+') as f:
    for line in f.readlines():
        if 'aggregating' in line:
            id = line.split(' ')[0]
            fitness_start = line.find('[')
            fitness = line[fitness_start + 2:-3].split(', ')
            genome_fitnesses[id] = fitness
        elif 'modules' in line:
            if 'loading' in line:
                continue

            fitness = line.split(' ')[-1]
            id_start = line.find('[')
            id_end = line.find(']')

            id = line[id_start+1:id_end].split(', ')
            mod_fitness[fitness] = id
        if line[:-1] == 'Step ended':
            break

print(genome_fitnesses)
for fit, ids in mod_fitness.items():
    for id in ids:
        if fit[:-1] in genome_fitnesses[id]:
            print('yay')
        else:
            print('nay')
            print(id, fit[:-1])