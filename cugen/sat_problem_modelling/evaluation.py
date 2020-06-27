import numpy


def evaluate_individual_fitness(individual: numpy.ndarray, formula: numpy.ndarray) -> numpy.ndarray:
    """
    Evaluates the fitness of an individual as a function of how many clauses in the formula it satisfies. For example
    if an individual satisfies half of the clauses, its fitness will be 0.5

    :param individual: The individual for which we compute the fitness
    :param formula: The CNF formula to satisfy
    :return: The value of the fitness of an individual, or the ratio of clauses of the formula it satisfies
    """
    return numpy.mean(numpy.any(individual == formula, axis=1), dtype=numpy.float16)


def evaluate_population(population: numpy.ndarray, formula: numpy.ndarray) -> numpy.ndarray:
    """
    Evaluates the fitness of every individual in a population and returns the fitness values in the order of the
    individuals in the population matrix.

    :param population: A matrix where each row represents an individual to be evaluated
    :param formula: The CNF formula to satisfy
    :return: The values of fitness of each individual in the input population as an array
    """
    return numpy.mean(numpy.any(population[:, numpy.newaxis] == formula, axis=2), axis=1, dtype=numpy.float16)
