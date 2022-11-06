import os
import json
from collections import namedtuple
from utils.argutils import print_args
from encoder.train import train
from pathlib import Path
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("cfg", type=str)
    args = parser.parse_args()
    
    with open(args.cfg, 'r') as f:
        cfgs = json.load(f, object_hook=lambda d: namedtuple('x', d.keys())(*d.values()))
    
    os.makedirs(cfgs.models_dir, exist_ok=True)
    
    train(cfgs)
    