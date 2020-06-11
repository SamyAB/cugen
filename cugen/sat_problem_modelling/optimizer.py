import cupy

from cugen.sat_problem_modelling.crossover import population_binary_crossover
from cugen.sat_problem_modelling.evaluation import evaluate_population
from cugen.sat_problem_modelling.mutation import population_mutation
from cugen.sat_problem_modelling.selection import select_individuals


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
    Finds the best solution to the input formula within the number of generation given as input.

    The optimizer stops, either if it finds the perfect individual, or if the number of generation equals the maximum
    number of generations

    :param formula: The formula to solve
    :param maximum_number_of_generations: The maximum of number of generations before the stopping of the optimizer
    :param population_size: The number of individuals in each generation
    :return: The best individual found
    """
    population = generate_random_first_generation(population_size, formula.shape[1])
    population_fitness = evaluate_population(population, formula)
    best_fitness = cupy.max(population_fitness)
    best_individual = population_fitness[cupy.argmax(best_fitness)]

    if best_fitness == 1.:
        return best_individual

    for _ in range(maximum_number_of_generations):
        breading_population = select_individuals(population, population_fitness)
        population = population_mutation(population_binary_crossover(breading_population, population_size))
        population_fitness = evaluate_population(population, formula)

        if cupy.max(population_fitness) > best_fitness:
            best_fitness = cupy.max(population_fitness)
            best_individual = population[cupy.argmax(population_fitness)]

            if best_fitness == 1.:
                break

    return best_individual
