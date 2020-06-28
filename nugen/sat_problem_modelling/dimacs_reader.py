from pathlib import Path

import numpy


def transform_dimacs_clause_to_nugen_clause(dimacs_clause: str, number_of_literals: int) -> numpy.array:
    """
    Transform a clause line from the DIMACS format to a clause in the nugen SAT format

    :param dimacs_clause: Line representing a clause in a DIMACS file
    :param number_of_literals: The number of literals in the formula
    :return: The clause in the nugen SAT format
    """
    nugen_clause = numpy.array([numpy.nan] * number_of_literals)
    signed_literals = dimacs_clause.split()[:-1]

    for literal in signed_literals:
        literal_as_integer = int(literal)
        literal_as_index = int(numpy.absolute(literal_as_integer) - 1)

        if numpy.isnan(nugen_clause[literal_as_index]):
            nugen_clause[literal_as_index] = 0 if literal_as_integer < 0 else 1
        elif literal_as_integer != nugen_clause[literal_as_index]:
            nugen_clause[literal_as_index] = -2

    nugen_clause[nugen_clause == -2] = numpy.nan

    return nugen_clause


def read_dimacs_file(dimacs_file_path: Path) -> numpy.ndarray:
    """
    Given a path to DIMACS CNF file, this function returns a formula in the nugen SAT format

    :param dimacs_file_path: The path to the DIMACS
    :return: The CNF formula in the nugen SAT format
    """
    with dimacs_file_path.open() as dimacs_file:
        line = dimacs_file.readline()
        while line[0] == 'c':
            line = dimacs_file.readline()

        number_of_literals = int(line.split()[2])

        clauses = []
        for line in dimacs_file.readlines():
            if line[0] == 'c':
                continue
            if line[0] == '%':
                return numpy.array(clauses)
            clauses.append(transform_dimacs_clause_to_nugen_clause(line, number_of_literals))

    return numpy.array(clauses)
