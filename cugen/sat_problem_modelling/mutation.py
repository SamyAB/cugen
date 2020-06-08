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
