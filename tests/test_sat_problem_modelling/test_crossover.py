from unittest.mock import patch

import cupy
import pytest

from cugen.sat_problem_modelling.crossover import uniform_binary_crossover, population_binary_crossover

TESTED_MODULE = 'cugen.sat_problem_modelling.crossover'


@patch(f'{TESTED_MODULE}.cupy.random.uniform')
def test_uniform_binary_crossover_generates_a_new_individual_combining_the_two_parents(mock_random_uniform):
    # Given
    first_parent_individual = cupy.array([1, 0, 1, 0, 1, 0])
    second_parent_individual = cupy.array([0, 1, 0, 1, 0, 1])

    mock_random_uniform.return_value = cupy.array([0.4, 0.5, 0.6, 0.7, 0.3, 0.2])

    expected_child_individual = cupy.array([1, 0, 0, 1, 1, 0])

    # When
    child_individual = uniform_binary_crossover(first_parent_individual, second_parent_individual)

    # Then
    cupy.testing.assert_array_equal(child_individual, expected_child_individual)


@pytest.mark.parametrize("next_generation_size", [4, 5, 6, 7])
def test_crossover_population_yields_a_population_with_input_size_when_the_number_of_parents_is_even(
        next_generation_size
):
    # Given
    population = cupy.array([
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ])

    # When
    new_generation = population_binary_crossover(population, next_generation_size)

    # Then
    assert new_generation.shape[0] == next_generation_size


@pytest.mark.parametrize("next_generation_size", [4, 5, 6, 7])
def test_crossover_population_yields_a_population_with_input_size_when_the_number_of_parents_is_odd(
        next_generation_size
):
    # Given
    population = cupy.array([
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [1, 1, 0, 0],
        [0, 0, 1, 1],
        [1, 1, 1, 1],
    ])

    # When
    new_generation = population_binary_crossover(population, next_generation_size)

    # Then
    assert new_generation.shape[0] == next_generation_size
