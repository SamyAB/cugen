import cupy


def evaluate_individual_fitness(individual: cupy.ndarray, formula: cupy.ndarray) -> cupy.ndarray:
    """
    Evaluates the fitness of an individual as a function of how many clauses in the formula it satisfies. For example
    if an individual satisfies half of the clauses, its fitness will be 0.5

    :param individual: The individual for which we compute the fitness
    :param formula: The CNF formula to satisfy
    :return: The value of the fitness of an individual, or the ratio of clauses of the formula it satisfies
    """
    return cupy.mean(cupy.any(individual == formula, axis=1), dtype=cupy.float16)
