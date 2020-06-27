from unittest.mock import patch

import numpy

from cugen.sat_problem_modelling.mutation import individual_mutation, population_mutation

TESTED_MODULE = 'cugen.sat_problem_modelling.mutation'


@patch(f'{TESTED_MODULE}.numpy.random.uniform', return_value=numpy.array(0.001))
@patch(f'{TESTED_MODULE}.numpy.random.randint', return_value=numpy.array(0))
def test_individual_mutation_should_change_change_one_literal_value_in_the_individual(mock_uniform, mock_randint):
    # Given
    individual = numpy.array([0, 0, 0, 0])
    mutation_probability = .25

    expected_mutated_individual = numpy.array([1, 0, 0, 0])

    # When
    mutated_individual = individual_mutation(individual, mutation_probability)

    # Then
    numpy.testing.assert_array_equal(mutated_individual, expected_mutated_individual)


@patch(f'{TESTED_MODULE}.numpy.random.uniform', return_value=0.6)
def test_individual_mutation_returns_the_input_individual_if_random_value_is_bellow_mutation_probability(_):
    # Given
    input_individual = numpy.array([1, 2, 3, 4])
    mutation_probability = .25

    # When
    mutated_individual = individual_mutation(input_individual, mutation_probability)

    # Then
    assert mutated_individual is input_individual


@patch(f'{TESTED_MODULE}.numpy.random.uniform')
@patch(f'{TESTED_MODULE}.numpy.random.randint', return_value=numpy.array(2))
def test_population_mutation_should_apply_the_mutation_on_all_individuals_of_the_population(_, mock_uniform):
    # Given
    population = numpy.array([
        [0, 0, 0, 1],
        [0, 1, 0, 0],
        [1, 0, 1, 1],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
    ])
    mutation_probability = .25

    mock_uniform.side_effect = [numpy.array(0.001), numpy.array(1), numpy.array(1), numpy.array(1), numpy.array(0.001)]

    expected_mutated_population = numpy.array([
        [0, 0, 1, 1],
        [0, 1, 0, 0],
        [1, 0, 1, 1],
        [0, 1, 0, 1],
        [1, 0, 0, 0],
    ])

    # When
    mutated_population = population_mutation(population, mutation_probability)

    # Then
    numpy.testing.assert_array_equal(mutated_population, expected_mutated_population)
