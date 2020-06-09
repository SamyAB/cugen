from unittest.mock import patch

import cupy

from cugen.sat_problem_modelling.crossover import uniform_binary_crossover

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
