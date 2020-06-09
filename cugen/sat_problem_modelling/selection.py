import cupy

SELECTION_RATIO = 0.5


def select_individuals(population, population_fitness):
    """
    Selects randomly, with a weight, a number of individuals.

    The population and the population fitness must be indexed in the same way: The first individual in the population
    has the first fitness in the population fitness

    :param population: The population from which the individuals are selected
    :param population_fitness: The fitness of each individual in the population, used as weights in the random choice
    :return: A sample of the population with a size <= size(population) * SELECTION_RATIO
    """
    population_size = population.shape[0]
    target_population_size = population_size * SELECTION_RATIO
    selected_individuals = cupy.random.choice(population_size, size=target_population_size, p=population_fitness)
    selected_individuals_with_no_duplication = cupy.unique(selected_individuals)

    return population[selected_individuals_with_no_duplication, :]
