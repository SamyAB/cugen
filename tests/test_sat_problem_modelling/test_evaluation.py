import numpy
import pytest

from nugen.sat_problem_modelling.evaluation import evaluate_individual_fitness, evaluate_population


@pytest.fixture()
def fixture_formula_with_one_literal_clauses():
    """
    This formula could be seen as
    f = B and not C and D and not A
    """
    formula = numpy.array([
        [numpy.nan, 1, numpy.nan, numpy.nan],
        [numpy.nan, numpy.nan, 0, numpy.nan],
        [numpy.nan, numpy.nan, numpy.nan, 1],
        [0, numpy.nan, numpy.nan, numpy.nan],
    ])
    return formula


def test_evaluate_individual_fitness_returns_perfect_fitness_if_the_individual_satisfies_all_the_formula_clauses(
        fixture_formula_with_one_literal_clauses
):
    # Given
    individual = numpy.array([0, 1, 0, 1])

    prefect_fitness = numpy.array([1.])

    # When
    individual_fitness = evaluate_individual_fitness(individual, fixture_formula_with_one_literal_clauses)

    # Then
    assert individual_fitness == prefect_fitness


def test_evaluate_individual_fitness_returns_null_fitness_if_the_individual_does_not_satisfy_any_formula_clause(
        fixture_formula_with_one_literal_clauses
):
    # Given
    individual = numpy.array([1, 0, 1, 0])

    null_fitness = numpy.array([0.])

    # When
    individual_fitness = evaluate_individual_fitness(individual, fixture_formula_with_one_literal_clauses)

    # Then
    assert individual_fitness == null_fitness


def test_evaluate_individual_fitness_returns_half_when_individual_satisfies_half_the_formula_clauses(
        fixture_formula_with_one_literal_clauses
):
    # Given
    individual = numpy.array([0, 1, 1, 0])

    half = numpy.array([0.5])

    # When
    individual_fitness = evaluate_individual_fitness(individual, fixture_formula_with_one_literal_clauses)

    # Then
    assert individual_fitness == half


def test_evaluate_population_should_return_the_list_of_fitness_values_for_the_spectrum_of_individuals_used(
        fixture_formula_with_one_literal_clauses
):
    # Given
    population = numpy.array([
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [0, 1, 1, 0]
    ])

    expected_population_fitness_values = numpy.array([1., 0., 0.5])

    # When
    population_fitness_values = evaluate_population(population, fixture_formula_with_one_literal_clauses)

    # Then
    numpy.testing.assert_array_equal(population_fitness_values, expected_population_fitness_values)


def test_evaluate_population_accurately_calculates_the_fitness_of_each_individual_with_regard_to_the_formula():
    # Given
    population = numpy.array([
        [1, 0, 1, 0],
        [1, 0, 0, 0],
        [1, 0, 1, 1],
    ])
    formula = numpy.array([
        [1, numpy.nan, numpy.nan, numpy.nan],
        [numpy.nan, 1, numpy.nan, numpy.nan],
        [numpy.nan, numpy.nan, 1, numpy.nan],
        [numpy.nan, numpy.nan, numpy.nan, 1],
    ])

    expected_population_fitness = numpy.array([0.5, 0.25, 0.75])

    # When
    population_fitness = evaluate_population(population, formula)

    # Then
    numpy.testing.assert_array_equal(population_fitness, expected_population_fitness)
