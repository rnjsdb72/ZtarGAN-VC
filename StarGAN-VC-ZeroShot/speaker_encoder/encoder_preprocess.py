import json
from collections import namedtuple
from pathlib import Path
import argparse
import pickle


if __name__ == "__main__":
    class MyFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
        pass
    
    parser = argparse.ArgumentParser()
    parser.add_argument("cfg", type=str)
    args = parser.parse_args()
    
    with open(args.cfg, 'r') as f:
        cfgs = json.load(f, object_hook=lambda d: namedtuple('x', d.keys())(*d.values()))

    os.makedirs(cfgs.out_dir, exist_ok=True)
    
    train_files = os.listdir(cfgs.datasets_root+'/mel_train')
    test_files = os.listdir(cfgs.datasets_root+'/mel_test')
    files = train_files + test_files
    spk_dict = dict()
    for file in tqdm(files):
        spk = file.split('/')[-1].split('_')[0]
        if spk_dict.get(spk):
            spk_dict[spk].append(file)
        else:
            spk_dict[spk] = [file]
        
    sources_fpath = cfgs.out_dir.joinpath("_sources.txt")
    
    sources_file = sources_fpath.open("w")
    for speaker in spk_dict.keys():
        for file in spk_dict[speaker]:
            sources_file.write("%s,%s\n" % (file, file))
    
    sources_file.close()