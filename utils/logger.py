import logging
from variables import config as cf
from utils.file import create_folder_if_not_exist


def get_logger():
    create_folder_if_not_exist('logs')
    log_file = f'logs/{cf.config.model_name}.log'
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)

    formatter = logging.Formatter('%(message)s')

    # Log into file
    handler = logging.FileHandler(log_file)
    handler.setLevel(logging.INFO)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # Log into terminal
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    console.setLevel(logging.INFO)
    logger.addHandler(console)

    return logger


def clean_logger():
    logger = logging.getLogger(__name__)
    logger_handlers_len = len(logger.handlers)

    for i in range(logger_handlers_len):
        logger.removeHandler(logger.handlers[0])
