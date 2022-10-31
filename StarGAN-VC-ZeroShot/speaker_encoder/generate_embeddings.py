from encoder import inference as encoder
from pathlib import Path
import json
from collections import namedtuple
import numpy as np
import librosa
import argparse
import torch
import sys
import os
import glob


if __name__ == '__main__':
    # Info & args
    parser = argparse.ArgumentParser()
    parser.add_argument("cfg", type=str)
    args = parser.parse_args()
    
    with open(args.cfg, 'r') as f:
        args = json.load(f, object_hook=lambda d: namedtuple('x', d.keys())(*d.values()))

    # Load the models one by one.
    print("Preparing the encoder...")
    encoder.load_model(args.enc_model_fpath, args)
    print("Insert the wav file name...")
    try:
        # Get the reference audio filepath

        for filename in glob.glob(os.path.join(args.audio_fpath, '*.wav')):
            mel = torch.tensor(np.load(filename))

        # Then we derive the embedding. There are many functions and parameters that the
        # speaker encoder interfaces. These are mostly for in-depth research. You will typically
        # only use this function (with its default parameters):
            embed = encoder.embed_utterance(mel, args)
            embed_path = args.embed_fpath / \
                filename.split('/')[-1]
            np.save(embed_path, embed)
            print("Created the embeddings")

    except Exception as e:
        print("Caught exception: %s" % repr(e))
        print("Restarting\n")
