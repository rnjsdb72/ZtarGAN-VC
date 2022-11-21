import os
from glob import glob
import json
from collections import namedtuple
import argparse
from multiprocessing import cpu_count
from solver import Solver
from data_loader import get_loader, TestDataset
from torch.backends import cudnn
from utils import arg_parse


def str2bool(v):
    return v.lower() in 'true'


def main(config, speakers):
    # For fast training.
    cudnn.benchmark = True

    # Create directories if not exist.
    if not os.path.exists(config.directories.log_dir):
        os.makedirs(config.directories.log_dir)
    if not os.path.exists(config.directories.model_save_dir):
        os.makedirs(config.directories.model_save_dir)
    if not os.path.exists(config.directories.sample_dir):
        os.makedirs(config.directories.sample_dir)

    # TODO: remove hard coding of 'test' speakers
    src_spk = config.src_spk
    trg_spk = config.trg_spk

    num_workers_ = config.miscellaneous.num_workers if config.miscellaneous.num_workers is not None else cpu_count()
    
    print('Get DataLoader!')
    # Data loader.
    train_loader = get_loader(speakers, config.speaker_encoder, config.directories.train_data_dir, config.train.batch_size, 'train', num_workers=num_workers_, prefix=config.prefix)
    # TODO: currently only used to output a sample whilst training
    test_loader = TestDataset(speakers, config.speaker_encoder, config.directories.test_data_dir, config.directories.wav_dir, src_spk=src_spk, trg_spk=trg_spk)

    # Solver for training and testing StarGAN.
    solver = Solver(train_loader, test_loader, speakers, config.speaker_encoder.config.model.model_embedding_size, config)

    if config.miscellaneous.mode == 'train':
        solver.train()


if __name__ == '__main__':
    args = arg_parse()
    with open(args.cfgs, 'r') as f:
        config = json.load(f, object_hook=lambda d: namedtuple('x', d.keys())(*d.values()))

    # no. of spks
    speakers = list(set(map(lambda x: x.split('/')[-1].split('_')[0], glob(config.directories.train_data_dir+'/*'))))

    if len(speakers) < 2:
        raise RuntimeError("Need at least 2 speakers to convert audio.")

    main(config, speakers)