import json
from dataclasses import asdict
from pathlib import Path
import os
from glob import glob
from tqdm import tqdm
import pickle
import soundfile as sf
import librosa
import numpy as np
from tqdm import tqdm
import argparse

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('cfgs', type=str)
    args = parser.parse_args()
    return args

root = '../data/train/'

def wav_to_mel(self,wav):
    spec = librosa.feature.melspectrogram(y=wav,
                                    sr= preprocessing_config['audio']['sampling_rate'], 
                                        n_fft=2048, 
                                        hop_length=preprocessing_config['stft']['hop_length'], 
                                        win_length=preprocessing_config['stft']['win_length'], 
                                        window='hann', 
                                        fmin = preprocessing_config['mel']['mel_fmin'],
                                        fmax = preprocessing_config['mel']['mel_fmax'],
                                        center=True, 
                                        pad_mode='reflect', 
                                        power=2.0,
                                    n_mels=128)
    return spec
class make_mel:
    def __init__(self, root = root):
        self.root = root
        #self.name = root.name

    def track_mel_path(self,wf_):
        wf_path = os.path.join(self.root,wf_)
        for env in tqdm(os.listdir(wf_path)):
            now_env = os.path.join(self.root,env)
            for days in os.listdir(now_env):
                now_days = os.path.join(days)
                path_1 = os.path.join(self.root,'mel')
                path_2 = os.path.join(path_1 , env)
                path_3 = os.path.join(path_2 , now_days)
                os.makedirs(path_3, exist_ok=True)
                for wf in os.listdir(now_days):
                    mels = wav_to_mel(wf)
                    name = wf.split('.')[0]
                    with open(f'{os.path.join(path_3,name)}.npy', 'wb') as f:
                        np.save(mels)


if __name__ == "__main__":
    args = arg_parse()
    with open(args.cfg, 'r') as f:
        cfgs = json.load(f)
    preprocessing_config = cfgs['preprocess']
    model_config = cfgs['model']
    train_config = cfgs['train']
    configs = (preprocessing_config)
    ms = make_mel(root)
    ms.track_mel_path()
