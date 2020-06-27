from unittest.mock import patch, call

import numpy
import pytest

from cugen.sat_problem_modelling.dimacs_reader import transform_dimacs_clause_to_cugen_clause, read_dimacs_file

TESTED_MODULE = 'cugen.sat_problem_modelling.dimacs_reader'


def test_transform_dimacs_clause_to_cugen_clause_transforms_correctly_dimacs_clause_to_nan_0_and_1_array():
    # Given
    dimacs_clause = "4 -8 9 0\n"
    number_of_literals = 10

    expected_cugen_clause = numpy.array([numpy.nan, numpy.nan, numpy.nan, 1, numpy.nan,
                                        numpy.nan, numpy.nan, 0, 1, numpy.nan])

    # When
    cugen_clause = transform_dimacs_clause_to_cugen_clause(dimacs_clause, number_of_literals)

    # Then
    numpy.testing.assert_array_equal(cugen_clause, expected_cugen_clause)


def test_transform_dimacs_clause_to_cugen_clause_transforms_contradicting_literals_to_nan():
    # Given
    dimacs_clause_with_contradicting_literals = "1 -1 3 4 0\n"

    expected_cugen_clause_with_nans = numpy.array([numpy.nan, numpy.nan, 1, 1])

    # When
    cugen_clause_with_nans = transform_dimacs_clause_to_cugen_clause(dimacs_clause_with_contradicting_literals,
                                                                     number_of_literals=4)

    # Then
    numpy.testing.assert_array_equal(cugen_clause_with_nans, expected_cugen_clause_with_nans)


def test_transform_dimacs_clause_to_cugen_clause_transforms_contradicting_literals_to_nan_in_case_of_odd_number_of_apparitions():
    # Given
    clause_with_contradictions = "1 -1 1 2 0\n"

    expected_cugen_clause_with_nans = numpy.array([numpy.nan, 1])

    # When
    cugen_clause_with_nans = transform_dimacs_clause_to_cugen_clause(clause_with_contradictions, number_of_literals=2)

    # Then
    numpy.testing.assert_array_equal(cugen_clause_with_nans, expected_cugen_clause_with_nans)


def test_transform_dimacs_clause_to_cugen_clause_raises_exception_when_literal_is_out_of_bounds():
    # Given
    clause_with_literal_out_of_bounds = '42 -246 528 0\n'
    number_of_literals = 527

    # Then
    with pytest.raises(IndexError):
        # When
        _ = transform_dimacs_clause_to_cugen_clause(clause_with_literal_out_of_bounds, number_of_literals)


@patch(f'{TESTED_MODULE}.transform_dimacs_clause_to_cugen_clause', return_value=numpy.array([1, 2, 3]))
def test_read_dimacs_file_finds_the_right_number_of_literals_and_loops_over_clauses_to_transform_them(
        mock_transform_clause, tmp_path
):
    # Given
    dimacs_content = "c This Formular is generated by mcnf\n" + \
                     "c\n" + \
                     "c    horn? no\n" + \
                     "c    forced? no\n" + \
                     "c    mixed sat? no\n" + \
                     "c    clause length = 3\n" + \
                     "c\n" + \
                     "p cnf 3 4\n" + \
                     " 1 2 3 0\n" + \
                     "-1 -2 -3 0\n" + \
                     "1 2 -3 0\n" + \
                     "-1 2 -3 0\n" + \
                     "%\n" + \
                     "0\n"
    dimacs_file_path = tmp_path / "cnf_formula.cnf"
    dimacs_file_path.write_text(dimacs_content)

    expected_transform_clause_calls = [
        call(" 1 2 3 0\n", 3),
        call("-1 -2 -3 0\n", 3),
        call("1 2 -3 0\n", 3),
        call("-1 2 -3 0\n", 3),
    ]

    # When
    _ = read_dimacs_file(dimacs_file_path)

    # Then
    mock_transform_clause.assert_has_calls(expected_transform_clause_calls)


@patch(f'{TESTED_MODULE}.transform_dimacs_clause_to_cugen_clause', return_value=numpy.array([1, 2, 3]))
def test_read_dimacs_file_ignores_all_lines_starting_with_c_not_only_at_the_start(
        mock_transform_clause, tmp_path
):
    # Given
    dimacs_content = "c This Formular is generated by mcnf\n" + \
                     "c\n" + \
                     "c    horn? no\n" + \
                     "c    forced? no\n" + \
                     "c    mixed sat? no\n" + \
                     "c    clause length = 3\n" + \
                     "c\n" + \
                     "p cnf 3 4\n" + \
                     "c This should be ignored\n" + \
                     " 1 2 3 0\n" + \
                     "-1 -2 -3 0\n" + \
                     "1 2 -3 0\n" + \
                     "-1 2 -3 0\n" + \
                     "%\n" + \
                     "0\n"
    dimacs_file_path = tmp_path / "cnf_formula.cnf"
    dimacs_file_path.write_text(dimacs_content)

    expected_transform_clause_calls = [
        call(" 1 2 3 0\n", 3),
        call("-1 -2 -3 0\n", 3),
        call("1 2 -3 0\n", 3),
        call("-1 2 -3 0\n", 3),
    ]

    # When
    _ = read_dimacs_file(dimacs_file_path)

    # Then
    assert call('c This should be ignored\n', 3) not in mock_transform_clause.call_args_list


def test_read_dimacs_files_returns_the_cugen_formula_even_if_the_dimacs_file_does_not_contain_the_percentage(tmp_path):
    # Given
    dimacs_content = "c    forced? no\n" + \
                     "c    mixed sat? no\n" + \
                     "c\n" + \
                     "p cnf 2 4\n" + \
                     "-1 0\n" + \
                     "1 0\n" + \
                     "-1 2 0\n"
    dimacs_file_path = tmp_path / "cnf_formula.cnf"
    dimacs_file_path.write_text(dimacs_content)

    expected_cugen_formula = numpy.array([
        [0, numpy.nan],
        [1, numpy.nan],
        [0, 1]
    ])

    # When
    cugen_formula = read_dimacs_file(dimacs_file_path)

    # Then
    numpy.testing.assert_array_equal(cugen_formula, expected_cugen_formula)
