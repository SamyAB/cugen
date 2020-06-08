from unittest.mock import patch

import cupy

from cugen.sat_problem_modelling.mutation import individual_mutation

TESTED_MODULE = 'cugen.sat_problem_modelling.mutation'


@patch(f'{TESTED_MODULE}.cupy.random.uniform', return_value=cupy.array(0.001))
@patch(f'{TESTED_MODULE}.cupy.random.randint', return_value=cupy.array(0))
def test_individual_mutation_should_change_change_one_literal_value_in_the_individual(mock_uniform, mock_randint):
    # Given
    individual = cupy.array([0, 0, 0, 0])

    expected_mutated_individual = cupy.array([1, 0, 0, 0])

    # When
    mutated_individual = individual_mutation(individual)

    # Then
    cupy.testing.assert_array_equal(mutated_individual, expected_mutated_individual)


@patch(f'{TESTED_MODULE}.cupy.random.uniform', return_value=0.6)
def test_individual_mutation_returns_the_input_individual_if_random_value_is_bellow_mutation_probability(_):
    # Given
    input_individual = cupy.array([1, 2, 3, 4])

    # When
    mutated_individual = individual_mutation(input_individual)

    # Then
    assert mutated_individual is input_individual
