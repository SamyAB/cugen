from unittest.mock import patch, call

import cupy

from cugen.sat_problem_modelling.optimizer import generate_random_first_generation, optimize

TESTED_MODULE = 'cugen.sat_problem_modelling.optimizer'


@patch(f'{TESTED_MODULE}.generate_random_first_generation')
@patch(f'{TESTED_MODULE}.evaluate_population')
@patch(f'{TESTED_MODULE}.select_individuals')
@patch(f'{TESTED_MODULE}.population_binary_crossover')
@patch(f'{TESTED_MODULE}.population_mutation')
@patch(f'{TESTED_MODULE}.cupy.max')
@patch(f'{TESTED_MODULE}.cupy.argmax')
def test_optimize_should_call_the_right_functions_in_the_right_order(mock_argmax, mock_max, mock_mutation,
                                                                     mock_crossover, mock_selection,
                                                                     mock_evaluate_population,
                                                                     mock_generate_first_generation):
    # Given
    formula = cupy.array([
        [1, cupy.nan, cupy.nan],
        [0, cupy.nan, cupy.nan]
    ])
    number_of_iterations = 10
    population_size = 2
    selection_ratio = 0.5
    mutation_probability = 0.25

    mock_generate_first_generation.return_value = "Population"
    mock_evaluate_population.return_value = "Population evaluation array"
    mock_selection.return_value = "Selected population"
    mock_crossover.return_value = "Crossover population"
    mock_mutation.return_value = "Population"
    mock_max.return_value = 15
    mock_argmax.return_value = 0

    expected_evaluate_population_calls = [call("Population", formula)] * (number_of_iterations + 1)
    expected_selection_calls = [call("Population", "Population evaluation array",
                                     selection_ratio)] * number_of_iterations
    expected_crossover_calls = [call("Selected population", population_size)] * number_of_iterations
    expected_mutation_calls = [call("Crossover population", mutation_probability)]

    # When
    _ = optimize(formula, number_of_iterations, population_size, selection_ratio, mutation_probability)

    # Then
    mock_generate_first_generation.assert_called_once_with(population_size, formula.shape[1])
    mock_evaluate_population.has_calls(expected_evaluate_population_calls)
    mock_selection.assert_has_calls(expected_selection_calls)
    mock_crossover.assert_has_calls(expected_crossover_calls)
    mock_mutation.assert_has_calls(expected_mutation_calls)


@patch(f'{TESTED_MODULE}.evaluate_population')
@patch(f'{TESTED_MODULE}.select_individuals')
@patch(f'{TESTED_MODULE}.population_mutation')
@patch(f'{TESTED_MODULE}.population_binary_crossover')
def test_optimizer_stops_when_the_perfect_individual_has_been_found_in_a_generation(mock_crossover,
                                                                                    mock_mutation,
                                                                                    mock_selection,
                                                                                    mock_evaluation):
    # Given
    formula = cupy.array([
        [0, 0, 1],
        [cupy.nan, cupy.nan, 0]
    ])
    number_of_generations = 15
    population_size = 4
    selection_ratio = 0.5
    mutation_probability = .25

    mock_evaluation.side_effect = [cupy.array([0.5, 0.2, 0.3, 0.7]), cupy.array([1.0, 0.2, 0.3, 0.4])]

    # When
    _ = optimize(formula, number_of_generations, population_size, selection_ratio, mutation_probability)

    # Then
    mock_selection.assert_called_once()


@patch(f'{TESTED_MODULE}.evaluate_population', return_value=cupy.array([1.0, 0.2, 0.3]))
def test_optimize_does_not_start_optimization_if_there_is_a_perfect_individual_in_the_initial_generation(mock_evaluate):
    # Given
    formula = cupy.array([[1., 0., 1.]])
    number_of_generations = 10
    population_size = 3
    selection_ratio = 0.5
    mutation_probability = .25

    # When
    _ = optimize(formula, number_of_generations, population_size, selection_ratio, mutation_probability)

    # Then
    mock_evaluate.assert_called_once()


def test_generate_random_first_generation_returns_population_with_the_right_shape():
    # Given
    number_of_individuals = 100
    number_of_literals = 40

    # When
    initial_population = generate_random_first_generation(number_of_individuals, number_of_literals)

    # Then
    assert initial_population.shape == (number_of_individuals, number_of_literals)


def test_generate_random_first_generation_returns_individuals_with_values_that_are_either_0_or_1():
    # Given
    number_of_individuals = 100
    number_of_literals = 40

    # When
    population = generate_random_first_generation(number_of_individuals, number_of_literals)

    # Then
    assert cupy.all(population <= 1)
    assert cupy.all(population >= 0)
