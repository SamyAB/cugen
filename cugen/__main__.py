import logging
from pathlib import Path

import typer

from cugen.sat_problem_modelling.dimacs_reader import read_dimacs_file
from cugen.sat_problem_modelling.optimizer import optimize
from cugen.utils.logger import init_logger

app = typer.Typer()


@app.command()
def sat_solver(
        path_to_dimacs_file: Path, number_of_maximum_generations: int, number_of_individuals_in_population: int,
        mutation_probability: float, selection_ratio: float
):
    """
    Main command for the SAT solver. As of this version, only DIMACS cnf formulas can be read and solved

    :param path_to_dimacs_file: Path to the formula to solve
    :param number_of_maximum_generations: The number maximum number of iterations
    :param number_of_individuals_in_population: The number of individuals in a single iteration
    :param mutation_probability: The probability of mutation of each new individual
    :param selection_ratio: The proportion of the population that survives to breed a new generation
    """
    init_logger(path_to_dimacs_file.stem, number_of_individuals_in_population)

    logging.info('Reading DIMACS file')
    formula = read_dimacs_file(path_to_dimacs_file)

    logging.info('Start the optimization')
    best_individual = optimize(formula,
                               maximum_number_of_generations=number_of_maximum_generations,
                               mutation_probability=mutation_probability,
                               selection_ratio=selection_ratio,
                               population_size=number_of_individuals_in_population,
                               )
    print(best_individual)


if __name__ == "__main__":
    app()
