from unittest.mock import patch

import cupy

from cugen.sat_problem_modelling.selection import select_individuals

TESTED_MODULE = 'cugen.sat_problem_modelling.selection'


@patch(f'{TESTED_MODULE}.cupy.random.uniform', return_value=cupy.array([0.4, 0.2, 0.2]))
def test_select_individuals_returns_population_sample_according_to_the_fitness_of_all_individuals(mock_random_uniform):
    # Given
    population = cupy.array([
        [1, 0, 0, 0],
        [1, 0, 0, 1],
        [1, 0, 1, 0],
    ])
    population_fitness = cupy.array([0.2, 0.3, 0.4])
    selection_ratio = 2/3

    expected_selected_individuals = cupy.array([
        [1, 0, 1, 0],
        [1, 0, 0, 0]
    ])

    # When
    selected_individuals = select_individuals(population, population_fitness, selection_ratio)

    # Then
    mock_random_uniform.assert_called_once_with(size=population.shape[0])
    cupy.testing.assert_array_equal(selected_individuals, expected_selected_individuals)
