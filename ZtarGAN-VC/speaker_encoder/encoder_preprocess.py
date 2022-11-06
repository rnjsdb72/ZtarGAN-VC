import os
from tqdm import tqdm
import json
from collections import namedtuple
from pathlib import Path
import argparse
import pickle


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("cfg", type=str)
    args = parser.parse_args()
    
    with open(args.cfg, 'r') as f:
        cfgs = json.load(f, object_hook=lambda d: namedtuple('x', d.keys())(*d.values()))

    os.makedirs(cfgs.out_dir, exist_ok=True)
    
    train_files = os.listdir(cfgs.datasets_root+'/mel_train')
    test_files = os.listdir(cfgs.datasets_root+'/mel_test')
    
    print("Generating Train Data's Speaker Dictionary")
    spk_dict = dict()
    for file in tqdm(train_files):
        if file.endswith('npz'):
            continue
        spk = file.split('/')[-1].split('_')[0]
        if spk_dict.get(spk):
            spk_dict[spk].append(file)
        else:
            spk_dict[spk] = [file]
            
    with open(cfgs.out_dir+'/train_speakers.pkl', 'wb') as f:
        pickle.dump(list(spk_dict.keys()), f)
        
    print("Generating Train Data's Speaker Sources")
    for speaker in tqdm(spk_dict.keys()):
        sources_fpath = Path(cfgs.out_dir).joinpath(f"{speaker}_sources.txt")
        sources_file = sources_fpath.open("w")
        for file in spk_dict[speaker]:
            sources_file.write("%s,%s\n" % (file, file))
        sources_file.close()
        
    print("Generating Test Data's Speaker Dictionary")
    spk_dict = dict()
    for file in tqdm(test_files):
        spk = file.split('/')[-1].split('_')[0]
        if spk_dict.get(spk):
            spk_dict[spk].append(file)
        else:
            spk_dict[spk] = [file]
            
    with open(cfgs.out_dir+'/test_speakers.pkl', 'wb') as f:
        pickle.dump(list(spk_dict.keys()), f)