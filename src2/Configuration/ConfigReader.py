import configparser
from src2.Configuration.Configuration import config


def read(file: str):
    config_reader = configparser.ConfigParser()
    config_reader.read(file)
    print(dict(config_reader))
    extensions = config_reader['extensions']
    if extensions.getboolean('speciation'):
        # config.
        pass


read('base.ini')
