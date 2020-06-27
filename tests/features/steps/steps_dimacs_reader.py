from pathlib import Path

import cupy
from behave import given, when, then
from behave.runner import Context

from cugen.sat_problem_modelling.dimacs_reader import read_dimacs_file

RESOURCES_PATH = Path(__file__).parents[2] / 'resources'


@given('an input file generated from toughsat')
def input_toughsat_file(context: Context):
    context.path_to_dimacs_file = RESOURCES_PATH / 'toughsat_generated_formula.dimacs'


@when('the dimacs reader transforms the input file to a cugen formula')
def run_dimacs_reader(context: Context) -> None:
    context.formula = read_dimacs_file(context.path_to_dimacs_file)


@then('the cugen formula is the right formula')
def compare_read_formula_with_the_expected_one(context: Context) -> None:
    expected_formula = cupy.array([
        [cupy.nan, 0, 0, cupy.nan, cupy.nan, 1, 1, cupy.nan, 1],
        [cupy.nan, 0, 0, 0, cupy.nan, cupy.nan, cupy.nan, cupy.nan, cupy.nan],
        [cupy.nan, 1, 1, 1, cupy.nan, cupy.nan, cupy.nan, cupy.nan, 0],
        [0, cupy.nan, 0, cupy.nan, cupy.nan, cupy.nan, cupy.nan, cupy.nan, cupy.nan],
        [1, cupy.nan, 1, cupy.nan, cupy.nan, 1, cupy.nan, cupy.nan, cupy.nan],
        [cupy.nan, cupy.nan, cupy.nan, cupy.nan, cupy.nan, cupy.nan, cupy.nan, cupy.nan, cupy.nan],
        [1, cupy.nan, 0, cupy.nan, cupy.nan, 0, cupy.nan, cupy.nan, 0],
        [cupy.nan, 0, cupy.nan, cupy.nan, cupy.nan, 1, cupy.nan, 0, cupy.nan],
        [0, cupy.nan, 1, cupy.nan, cupy.nan, cupy.nan, cupy.nan, 1, cupy.nan],
        [cupy.nan, 1, cupy.nan, cupy.nan, cupy.nan, cupy.nan, 1, cupy.nan, cupy.nan],
        [cupy.nan, cupy.nan, 0, cupy.nan, 1, cupy.nan, cupy.nan, 0, cupy.nan],
    ])

    cupy.testing.assert_array_equal(context.formula, expected_formula)
