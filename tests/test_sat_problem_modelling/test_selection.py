from unittest.mock import patch

import cupy

from cugen.sat_problem_modelling.selection import select_individuals, SELECTION_RATIO

TESTED_MODULE = 'cugen.sat_problem_modelling.selection'


@patch(f'{TESTED_MODULE}.cupy.random.choice')
@patch(f'{TESTED_MODULE}.cupy.divide')
def test_select_individuals_calls_random_choice_with_the_right_parameters_by_transforming_the_fitness_to_probabilities(
        mock_divide, mock_random_choice
):
    # Given
    population = cupy.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ])
    population_fitness = cupy.array([1., 0.5, 0.5, 0.])

    selection_probabilities = cupy.array([0.2, 0.2, 0.4, 0.2])
    mock_divide.return_value = selection_probabilities
    mock_random_choice.return_value = [0, 1]

    expected_selected_individuals = cupy.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
    ])

    # When
    selected_individuals = select_individuals(population, population_fitness)

    # Then
    mock_random_choice.assert_called_once_with(population.shape[0], size=int(population.shape[0] * SELECTION_RATIO),
                                               p=selection_probabilities)
    cupy.testing.assert_array_equal(selected_individuals, expected_selected_individuals)


@patch(f'{TESTED_MODULE}.cupy.random.choice')
def test_select_individuals_does_not_select_an_individual_twice_as_long_as_it_is_not_already_duplicated_in_the_input(
        mock_random_choice
):
    """
    This test is here to ensure that even if random.choice returns duplicated values (because cupy does not implement
    yet replace=False), the selection won't duplicate an individual
    """
    # Given
    population = cupy.array([
        [1, 0, 1, 0],
        [1, 0, 0, 1],
        [0, 1, 0, 1],
        [0, 0, 0, 1],
        [0, 0, 1, 1],
    ])
    population_fitness = cupy.array([0.1, 0.5, 0.3, 0.2, 0.1])

    mock_random_choice.return_value = cupy.array([0, 0])

    expected_selected_individuals = cupy.array([[1, 0, 1, 0]])

    # When
    selected_individuals = select_individuals(population, population_fitness)

    # Then
    cupy.testing.assert_array_equal(selected_individuals, expected_selected_individuals)
