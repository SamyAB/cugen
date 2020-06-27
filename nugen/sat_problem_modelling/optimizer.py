import logging

import numpy

from cugen.sat_problem_modelling.crossover import population_binary_crossover
from cugen.sat_problem_modelling.evaluation import evaluate_population
from cugen.sat_problem_modelling.mutation import population_mutation
from cugen.sat_problem_modelling.selection import select_individuals


def generate_random_first_generation(population_size, number_of_genes_in_population) -> numpy.ndarray:
    """
    Generates a random first population

    :param population_size: The number of individuals in the population
    :param number_of_genes_in_population: The number of values representing each individuals
    :return: An initial population
    """
    return numpy.random.randint(low=0, high=2, size=(population_size, number_of_genes_in_population))


def optimize(formula: numpy.ndarray, maximum_number_of_generations: int, population_size: int, selection_ratio: float,
             mutation_probability: float) -> numpy.ndarray:
    """
    Finds the best solution to the input formula within the number of generation given as input.

    The optimizer stops, either if it finds the perfect individual, or if the number of generation equals the maximum
    number of generations

    :param formula: The formula to solve
    :param maximum_number_of_generations: The maximum of number of generations before the stopping of the optimizer
    :param population_size: The number of individuals in each generation
    :param selection_ratio: Between 0 and 1. The surviving ratio of a generation, that is then bred
    :param mutation_probability: Between 0 and 1. The probability that has an individual to mutate
    :return: The best individual found
    """
    population = generate_random_first_generation(population_size, formula.shape[1])
    population_fitness = evaluate_population(population, formula)
    best_fitness = numpy.max(population_fitness)
    best_individual = population[numpy.argmax(population_fitness)]
    logging.info(f'In the first generation the best fitness was {best_fitness * 100}%')

    if best_fitness == 1.:
        logging.info('Lucky ! The first generation contains the solution')
        return best_individual

    generation = 0
    for generation in range(maximum_number_of_generations):
        breeding_population = select_individuals(population, population_fitness, selection_ratio)
        population = population_mutation(population_binary_crossover(breeding_population, population_size),
                                         mutation_probability)
        population_fitness = evaluate_population(population, formula)

        if numpy.max(population_fitness) > best_fitness:
            best_fitness = numpy.max(population_fitness)
            best_individual = population[numpy.argmax(population_fitness)]
            logging.info(f'In the generation {generation + 1} new best individual has been found, '
                         f'and it satisfies {best_fitness * 100}% of the formula')

            if best_fitness == 1.:
                break
        else:
            if generation % 1000 == 0:
                logging.info(f'After {generation + 1}, the best individuals satisfies {best_fitness * 100}% '
                             f'of the formula')

    logging.info('Optimization ended.')
    logging.info(f'The best individual found after {generation + 1} generations, '
                 f'and it satisfies {best_fitness * 100}% of the formula')

    return best_individual
