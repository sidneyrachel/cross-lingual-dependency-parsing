from models.config import Config


def init_config(filename):
    global config
    config = Config(filename)
