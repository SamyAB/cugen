import logging


def init_logger():
    logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%d-%m-%Y %I:%M:%S', level=logging.INFO)
