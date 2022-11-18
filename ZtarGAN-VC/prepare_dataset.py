from ast import excepthandler
import os
import pickle
import numpy as np
import soundfile as sf
from numpy.random import RandomState
from utils import *
import json
from collections import namedtuple
from tqdm import tqdm

def move_wav():
    call_dir = '../../raw_data/raw_wav/call'
    random_dir = '../../raw_data/raw_wav/random'
    continuous_dir = '../../raw_data/raw_wav/continuous'
    common_dir = '../../raw_data/raw_wav/common'

    dir_list = [call_dir,random_dir,continuous_dir,common_dir]
    if not os.path.exists('../../raw_data/wav'):
        os.makedirs('../../raw_data/wav')
    for i in tqdm(dir_list):
        try:
            for j in os.listdir(i): # 각 날짜 출력
                path_ = os.path.join(i,j)       
                for k in os.listdir(path_):   # 날짜 안에 들어있는 id_list
                    spk_path = f'../../raw_data/wav/{k}'
                    if not os.path.exists(spk_path):
                        os.makedirs(spk_path)
                    wavs = os.listdir(os.path.join(path_,k))
                    for wav in wavs:
                        os.replace(os.path.join(os.path.join(path_, k), wav), os.path.join(spk_path, wav))
        except:
            pass

def wav_name():
    call_raw = '../../raw_data/wav'
    for dir in tqdm(os.listdir(call_raw)):
        path_ = os.path.join(call_raw,dir)
        ct = 1
        for wav in os.listdir(path_):
            os.rename(os.path.join(path_ ,wav),os.path.join(path_ ,f'{dir}_{ct}.wav'))
            ct+=1

if __name__ == "__main__":
    args = arg_parse()
    with open(args.cfgs, 'r') as f:
        cfgs = json.load(f, object_hook = lambda d: namedtuple('x', d.keys())(*d.values()))

    move_wav()
    print('Rename wav file as Format...')
    wav_name()