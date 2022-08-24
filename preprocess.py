import json
from collections import namedtuple

from utils.tools import arg_parse
from preprocessor.preprocessor import Preprocessor


if __name__ == "__main__":
    args = arg_parse()
    with open(args.cfg, 'r') as f:
        cfgs = json.load(f)
    config = cfgs['preprocess']
    preprocessor = Preprocessor(config)
    preprocessor.build_from_path()