"""
This package contains the modelling of the SAT problem.

Each potential solution to the SAT problem is called an individual. As such all the individuals are represented with
ordered (the literal in the first position is the first literal, the second is the second literal...),
fixed length arrays (len = number ov literals in the formula), where 0 means a False literal, and 1 a True literal

Example : A = True, B = False, C = True is represented as [1, 0, 1]

A formula is represented as a matrix of shape (NxM), where N is the maximum number of literals in a clause, and M is
the number of clauses of the formula.

Each row of the formula matrix represents a clause, which has the same literal order as the individuals (meaning
the first literal of a clause is the first literal of every individual, ...), and when a literal is not in clause
the value is represented by a NaN

Example : The formula (not A or B) and (not B or C) is represented as
F = [
        [0, 1, NaN],
        [NaN, 0, 1]
    ]

"""