import cupy
from behave import given, when, then
from behave.runner import Context

from cugen.sat_problem_modelling.optimizer import optimize


@given('a simple CNF formula')
def simple_formula(context: Context):
    context.formula = cupy.array([
        [1, 0, 1, 1],
        [cupy.nan, cupy.nan, 1, cupy.nan],
        [cupy.nan, 0, cupy.nan, cupy.nan],
        [1, cupy.nan, cupy.nan, cupy.nan],
        [cupy.nan, cupy.nan, cupy.nan, 1]
    ])


@when('the solver is run with set hyper parameters')
def run_the_solver_in_easy_mode(context: Context):
    context.best_individual = optimize(
        context.formula,
        maximum_number_of_generations=1000000,
        population_size=10,
        selection_ratio=0.5,
        mutation_probability=.25
    )


@then('the solver finds the perfect solution')
def the_best_individual_is_found(context: Context):
    cupy.testing.assert_array_equal(context.best_individual, cupy.array([1, 0, 1, 1]))
