import cupy

from cugen.sat_problem_modelling.dimacs_reader import transform_dimacs_clause_to_cugen_clause


def test_transform_dimacs_clause_to_cugen_clause_transforms_correctly_dimacs_clause_to_nan_0_and_1_array():
    # Given
    dimacs_clause = "4 -8 9 0"
    number_of_literals = 10

    expected_cugen_clause = cupy.array([cupy.nan, cupy.nan, cupy.nan, 1, cupy.nan,
                                        cupy.nan, cupy.nan, 0, 1, cupy.nan])

    # When
    cugen_clause = transform_dimacs_clause_to_cugen_clause(dimacs_clause, number_of_literals)

    # Then
    cupy.testing.assert_array_equal(cugen_clause, expected_cugen_clause)
