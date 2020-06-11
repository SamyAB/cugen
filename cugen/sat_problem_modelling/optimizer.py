import cupy


def generate_random_first_generation(population_size, number_of_genes_in_population) -> cupy.ndarray:
    """
    Generates a random first population

    :param population_size: The number of individuals in the population
    :param number_of_genes_in_population: The number of values representing each individuals
    :return: An initial population
    """
    return cupy.random.uniform(low=0, high=1, size=(population_size, number_of_genes_in_population))


def optimize(formula: cupy.ndarray, maximum_number_of_generations: int, population_size: int) -> cupy.ndarray:
    """

    :param formula:
    :param maximum_number_of_generations:
    :param population_size:
    :return:
    """
    pass
