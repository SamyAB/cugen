import logging


def init_logger(formula_name: str, number_of_individuals: int) -> None:
    logging.basicConfig(filename=f'numpy_{formula_name}_{number_of_individuals}.log', format='%(asctime)s %(message)s',
                        datefmt='%d-%m-%Y %I:%M:%S', level=logging.INFO)
