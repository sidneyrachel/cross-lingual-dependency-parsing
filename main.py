import sys
from models.dependency_parser import DependencyParser
from variables.config import init_config


if __name__ == '__main__':
    config_filename = sys.argv[1]
    init_config(f'configs/{config_filename}.json')

    parser = DependencyParser()
    parser.train()

    parser = DependencyParser()
    parser.load_model(postfix='best')
    parser.evaluate(set_name='dev')
    parser.evaluate(set_name='test')
