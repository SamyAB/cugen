import cupy


def uniform_binary_crossover(first_parent_individual: cupy.ndarray,
                             second_parent_individual: cupy.ndarray) -> cupy.ndarray:
    """
    Builds a new child by randomly choosing parts of the two input parents, mimicking biological reproduction

    :param first_parent_individual: The array representing the first parent
    :param second_parent_individual: The array representing the second parent
    :return: A new individual from the mixing of the two input individuals
    """
    return cupy.where(cupy.random.uniform() <= 0.5, first_parent_individual, second_parent_individual)
