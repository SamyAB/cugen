import cupy

MUTATION_PROBABILITY = .01


def individual_mutation(individual: cupy.ndarray) -> cupy.ndarray:
    """
    Applies a mutation on the input individual, if a random value (from a uniform distribution) is above the mutation
    probability threshold. The literal to be mutated is also selected randomly in the cas of a mutation.

    If the threshold is not met, the function returns the input individual

    :param individual: The individual to mutate
    :return: Either the input individual if the threshold is not met, or the mutated individual otherwise
    """
    if cupy.random.uniform() <= MUTATION_PROBABILITY:
        mutated_individual = individual.copy()
        element_to_mutate = cupy.random.randint(low=0, high=mutated_individual.shape[0])
        mutated_individual[element_to_mutate] = 1 - mutated_individual[element_to_mutate]
        return mutated_individual

    return individual


def population_mutation(population: cupy.ndarray) -> cupy.ndarray:
    """
    Runs the individual mutation for every individual in a given population

    Note : This function might be a bottleneck as it uses a python for loop, if anyone has an idea that would make
    go faster, please share it

    :param population: The set of individuals to mutate
    :return: The input population with individual randomly mutated
    """
    for individual_index, individual in enumerate(population):
        population[individual_index] = individual_mutation(individual)

    return population
