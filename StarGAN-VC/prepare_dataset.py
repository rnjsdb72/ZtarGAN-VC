from ast import excepthandler
import os
import pickle
import argparse
import numpy as np
import soundfile as sf
from numpy.random import RandomState
from utils import *
import json
from collections import namedtuple
from tqdm import tqdm

def make_spk_meta():

    dir_list = [cfgs.call_dir, cfgs.random_dir, cfgs.continuous_dir,cfgs.common_dir]
    id_sex_dict = dict()
    for i in dir_list:
        for j in os.listdir(i): # 각 날짜 출력
            path_ = os.path.join(i,j)       
            for k in os.listdir(path_):   # 날짜 안에 들어있는 id_list
                json_path = os.listdir(os.path.join(path_,k))[0] # id list 까지의 데이터
                with open(os.path.join(os.path.join(path_,k) ,json_path),'r') as f:
                    json_data = json.load(f)
                    sex = json_data['Speaker']['Gender'][0]
                id_sex_dict[k] = sex
    dict_pkl = dict()
    for key, value in id_sex_dict.items():
        dict_pkl[key] = (key,value)
    if not os.path.exists('./preprocessed_data'):
        os.makedirs('./preprocessed_data')
    with open('./preprocessed_data/spk_meta.pkl', 'wb') as f:
        pickle.dump(dict_pkl, f, pickle.HIGHEST_PROTOCOL)

def move_wav(cfgs):

    dir_list = [cfgs.call_dir,cfgs.random_dir,cfgs.continuous_dir,cfgs.common_dir]
    if not os.path.exists(cfgs.wav_dir):
        os.makedirs(f'{cfgs.wav_dir}/wav')
    for i in tqdm(dir_list):
        for j in os.listdir(i): # 각 날짜 출력
            path_ = os.path.join(i,j)       
            for k in os.listdir(path_):   # 날짜 안에 들어있는 id_list
                spk_path = f'./{cfgs.wav_dir}/wav/{k}'
                if not os.path.exists(spk_path):
                    os.makedirs(spk_path)
                wavs = os.listdir(os.path.join(path_,k))
                for wav in wavs:
                    os.replace(os.path.join(os.path.join(path_, k), wav), os.path.join(spk_path, wav))

def wav_name(cfgs):
    call_raw = cfgs.wav_dir
    for dir in tqdm(os.listdir(call_raw)):
        path_ = os.path.join(call_raw,dir)
        ct = 1
        for wav in os.listdir(path_):
            os.rename(os.path.join(path_ ,wav),os.path.join(path_ ,f'{dir}_{ct}.wav'))
            ct+=1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--call_dir', type=str, default="./raw_data/raw_wav/call")
    parser.add_argument('--random_dir', type=str, default="./raw_data/raw_wav/random")
    parser.add_argument('--continuous_dir', type=str, default="./raw_data/raw_wav/continuous")
    parser.add_argument('--common_dir', type=str, default="./raw_data/raw_wav/common")
    parser.add_argument('--wav_dir', type=str, default="./raw_data/wav")
    parser.add_argument('--spk_meta_dir', type=str, default="./preprocessed_data/spk_meta.pkl")
    cfgs = parser.parse_args()
    
    print('Make Speaker Meta Data...')
    make_spk_meta(cfgs)
    move_wav(cfgs)
    print('Rename wav file as Format...')
    wav_name(cfgs)