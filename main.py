from models.dependency_parser import DependencyParser


if __name__ == '__main__':
    parser = DependencyParser()
    parser.train()
    parser.parse_sentence('The big dog lives in its little house.')
    parser.save_model()

    parser = DependencyParser()
    parser.load_model()
    parser.parse_sentence('The big dog lives in its little house.')
