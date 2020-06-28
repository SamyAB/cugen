import subprocess
import sys
from pathlib import Path

import numpy
from behave import given, when, then
from behave.runner import Context

from nugen.sat_problem_modelling.optimizer import optimize

RESOURCES_PATH = Path(__file__).parents[2] / 'resources'


@given('a simple CNF formula')
def simple_formula(context: Context) -> None:
    context.formula = numpy.array([
        [1, 0, 1, 1],
        [numpy.nan, numpy.nan, 1, numpy.nan],
        [numpy.nan, 0, numpy.nan, numpy.nan],
        [1, numpy.nan, numpy.nan, numpy.nan],
        [numpy.nan, numpy.nan, numpy.nan, 1]
    ])


@given('a DIMACS CNF file')
def path_to_the_test_dimacs_file(context: Context) -> None:
    context.path_to_dimacs_file = RESOURCES_PATH / 'dimacs_test_file.cnf'


@when('the solver is run with set hyper parameters')
def run_the_solver_in_easy_mode(context: Context):
    context.best_individual = optimize(
        context.formula,
        maximum_number_of_generations=1000000,
        population_size=10,
        selection_ratio=0.5,
        mutation_probability=.25
    )


@when('the SAT solver command is run')
def run_sat_solver_command(context: Context) -> None:
    main_path = Path(__file__).parents[3] / 'nugen' / '__main__.py'
    maximum_number_of_generations = '1000000'
    population_size = '10'
    selection_ratio = '0.5'
    mutation_probability = '.25'

    sat_solver_process = subprocess.Popen(
        [sys.executable, main_path.as_posix(), context.path_to_dimacs_file.as_posix(), maximum_number_of_generations,
         population_size, mutation_probability, selection_ratio], stdout=subprocess.PIPE)
    process_output = sat_solver_process.communicate()[0]

    context.best_individual = eval(','.join(process_output.decode().strip().split()))


@then('the solver finds the perfect solution')
def the_best_individual_is_found(context: Context):
    numpy.testing.assert_array_equal(context.best_individual, numpy.array([1, 0, 1, 1]))
