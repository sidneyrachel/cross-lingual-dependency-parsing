import sys
from models.dependency_parser import DependencyParser
from variables.config import init_config
from utils.logger import get_logger, clean_logger


if __name__ == '__main__':
    config_filenames = sys.argv[1].split(',')

    for config_filename in config_filenames:
        init_config(f'configs/{config_filename}.json')
        logger = get_logger()

        parser = DependencyParser(logger=logger, is_train=True)
        parser.train()

        parser = DependencyParser(logger=logger, is_train=False)
        parser.load_model(postfix='best')
        parser.evaluate(set_name='dev')
        parser.evaluate(set_name='test')

        clean_logger()
