import json
from collections import namedtuple
import argparse

from preprocessor.preprocessor import Preprocessor
from preprocessor.generate_lab import generate_lab


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='train_config.json')
    parser.add_argument('--generate_lab', type=str, default='false')
    parser.add_argument('--preprocessor', type=str, default='true')
    args = parser.parse_args()
    
    with open(args.cfg, 'r') as f:
        cfgs = json.load(f)

    config = cfgs['preprocess']

    if args.generate_lab == 'true':
        generate_lab(config)

    if args.preprocessor == 'true':
        preprocessor = Preprocessor(config)
        preprocessor.build_from_path()