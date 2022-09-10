import os
import yaml
import argparse
from torch.backends import cudnn

from solver import Solver
from data_loader import get_loader
from preprocess import preprocess_data
from utils import Dict2Class


def main(config, args):
    # For fast training.
    cudnn.benchmark = True
    
    data_loader = get_loader(config)
    solver = Solver(data_loader, args, config)
    solver.train()
            
if __name__ == '__main__':
    args = arg_parse()
    with open(args.cfgs, 'r') as f:
        cfgs = json.load(f, object_hook = lambda d: namedtuple('x', d.keys())(*d.values()))

    if cfgs.train.model_type == 'F':
        cfgs.train.model_type = 'F'
        cfgs.model.dim_pit = cfgs.model.dim_con + cfgs.model.dim_pit # concatenate spectrogram and quantized pitch contour as the f0 converter input
    
    main(cfgs, args)
