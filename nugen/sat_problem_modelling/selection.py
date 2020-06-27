import numpy


def select_individuals(population: numpy.ndarray, population_fitness: numpy.ndarray, selection_ratio: float):
    """
    Selects randomly, with a weight, a number of individuals.

    The population and the population fitness must be indexed in the same way: The first individual in the population
    has the first fitness in the population fitness

    :param population: The population from which the individuals are selected
    :param population_fitness: The fitness of each individual in the population, used as weights in the random choice
    :param selection_ratio: Between 0 and 1. The ratio of input population individuals to return
    :return: A sample of the population with a size <= size(population) * SELECTION_RATIO
    """
    population_size = population.shape[0]
    target_population_size = int(population_size * selection_ratio)

    selection_score = numpy.random.uniform(size=population_size) * population_fitness

    population_sorted_by_selection_score = population[numpy.argsort(selection_score)[::-1]]

    return population_sorted_by_selection_score[:target_population_size]
