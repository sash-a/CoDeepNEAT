def CDN_pareto(individuals):
    # Sorting members on primary fitness
    # TODO need to define if bigger fitness is better or worse
    individuals.sort(key=lambda indv: indv.fitness[0], reverse=True)

    pf = [individuals[0]]  # pareto front populated with best individual in primary objective

    for indv in individuals[1:]:
        if indv.fitness[1] > pf[-1].fitness[1]:
            pf.append(indv)

    return pf
