from models.dependency_parser import DependencyParser
from variables.config import init_config


if __name__ == '__main__':
    init_config('configs/test.json')

    parser = DependencyParser()
    parser.train()
    parser.parse_sentence('The big dog lives in its little house.')
    parser.save_model()

    parser = DependencyParser()
    parser.load_model()
    parser.parse_sentence('The big dog lives in its little house.')
