from pathlib import Path

import cupy


def transform_dimacs_clause_to_cugen_clause(dimacs_clause: str, number_of_literals: int) -> cupy.array:
    """
    Transform a clause line from the DIMACS format to a clause in the cugen SAT format

    :param dimacs_clause: Line representing a clause in a DIMACS file
    :param number_of_literals: The number of literals in the formula
    :return: The clause in the cugen SAT format
    """
    cugen_clause = cupy.array([cupy.nan] * number_of_literals)
    signed_literals = dimacs_clause.split()[:-1]
    for literal in signed_literals:
        literal_as_integer = int(literal)
        cugen_clause[cupy.absolute(literal_as_integer) - 1] = 0 if literal_as_integer < 0 else 1

    return cugen_clause


def read_dimacs_file(dimacs_file_path: Path) -> cupy.ndarray:
    """
    Given a path to DIMACS CNF file, this function returns a formula in the cugen SAT format

    :param dimacs_file_path: The path to the DIMACS
    :return: The CNF formula in the cugen SAT format
    """
    with dimacs_file_path.open() as dimacs_file:
        line = dimacs_file.readline()
        while line[0] == 'c':
            line = dimacs_file.readline()

        number_of_literals = int(line.split()[2])

        clauses = []
        for line in dimacs_file.readlines():
            if line[0] == '%':
                return cupy.concatenate(clauses)
            clauses.append(transform_dimacs_clause_to_cugen_clause(line, number_of_literals))
