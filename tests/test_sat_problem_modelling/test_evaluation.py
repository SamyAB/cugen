import cupy
import pytest

from cugen.sat_problem_modelling.evaluation import evaluate_individual_fitness


@pytest.fixture()
def fixture_formula_with_one_literal_clauses():
    """
    This formula could be seen as
    f = B and not C and D and not A
    """
    formula = cupy.array([
        [cupy.nan, 1, cupy.nan, cupy.nan],
        [cupy.nan, cupy.nan, 0, cupy.nan],
        [cupy.nan, cupy.nan, cupy.nan, 1],
        [0, cupy.nan, cupy.nan, cupy.nan],
    ])
    return formula


def test_evaluate_individual_fitness_returns_perfect_fitness_if_the_individual_satisfies_all_the_formula_clauses(
        fixture_formula_with_one_literal_clauses
):
    # Given
    individual = cupy.array([0, 1, 0, 1])

    prefect_fitness = cupy.array([1.])

    # When
    individual_fitness = evaluate_individual_fitness(individual, fixture_formula_with_one_literal_clauses)

    # Then
    assert individual_fitness == prefect_fitness


def test_evaluate_individual_fitness_returns_null_fitness_if_the_individual_does_not_satisfy_any_formula_clause(
        fixture_formula_with_one_literal_clauses
):
    # Given
    individual = cupy.array([1, 0, 1, 0])

    null_fitness = cupy.array([0.])

    # When
    individual_fitness = evaluate_individual_fitness(individual, fixture_formula_with_one_literal_clauses)

    # Then
    assert individual_fitness == null_fitness


def test_evaluate_individual_fitness_returns_half_when_individual_satisfies_half_the_formula_clauses(
        fixture_formula_with_one_literal_clauses
):
    # Given
    individual = cupy.array([0, 1, 1, 0])

    half = cupy.array([0.5])

    # When
    individual_fitness = evaluate_individual_fitness(individual, fixture_formula_with_one_literal_clauses)

    # Then
    assert individual_fitness == half
