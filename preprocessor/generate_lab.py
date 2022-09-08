import pandas as pd
import json
import os
import re
from glob import glob
from tqdm import tqdm

from jamo import h2j,hangul_to_jamo,j2hcj
from g2pk import G2p
import jamotools

from utils.tools import arg_parse

def jamo_split(content):
    g2p=G2p()
    content=g2p(content)
    jamo=h2j(content).split(" ")

    return jamo

def generate_lab(cfg):
    # json 파일 수집
    print('Collect json files,,,')
    files = []
    if cfg['dataset'] == 'kss':
        for i, speaker in enumerate(tqdm(os.listdir(cfg['path']['raw_path']))):
            files = glob(os.path.join(cfg['path']['raw_path']+'/'+speaker+'/*.txt'))
    else: 
        for i, speaker in enumerate(tqdm(os.listdir(cfg['path']['raw_path']))):
            files = files + glob(os.path.join(cfg['path']['raw_path']+'/'+speaker+'/*.json'))
    
    # 음소 분리
    print('Split word to phoneme,,,')
    p_dct = {}
    if cfg['dataset'] == 'kss':
        filters = '([.,!?])'
        with open(files[0], 'r') as f:
            data = f
            while True:
                line = data.readline()
                if not line:
                    break
                text = line.split('|')[1]
                text = re.sub(re.compile(filters), '', text)
                words = text.split(" ")
                for word in words:
                    if not word in p_dct.keys():
                        p_dct[word] = " ".join(jamo_split(word)[0])
    else:
        for file in tqdm(files):
            with open(file, 'r') as f:
                data = json.load(f)
            text = data['전사정보']['TransLabelText']
            words = text.split(" ")
            for word in words:
                if not word in p_dct.keys():
                    p_dct[word] = " ".join(jamo_split(word)[0])
    
    # Lexicon 제작
    os.makedirs('./lexicon', exist_ok=True)
    with open("./lexicon/p_lexicon.txt", "w") as f:
        for k, v in p_dct.items():
            f.write(f"{k}\t{v}\n")
    print("Complete to make lexicon!")

    # lab file 제작
    print("Generate LAB file,,,")
    meta_lst = []
    if cfg['dataset'] == 'kss':
        with open(files[0], 'r') as f:
            data = f
            while True:
                line = data.readline()
                if not line:
                    break

                wav_path = line.split('|')[0].split('/')[-1]
                content = line.split('|')[1]
                text_path = wav_path.replace('wav', 'lab')
                file_path = os.listdir(cfg['path']['raw_path'])[0]
            
                with open(cfg['path']['raw_path'] + '/' + file_path + '/' + text_path, 'w') as t:
                        t.write(content)
    else:
        for file in tqdm(files):
            with open(file, 'r') as f:
                data = json.load(f)

            wav_path = data['파일정보']['FileName']
            content = data['전사정보']['TransLabelText']
            text_path = wav_path.replace('wav', 'lab')
            file_path = data['파일정보']['DirectoryPath']

            with open(cfg['path']['raw_path'] + file_path + '/' + text_path, 'w') as t:
                t.write(content)


if __name__ == "__main__":
    args = arg_parse()
    with open(args.cfg, 'r') as f:
        cfgs = json.load(f)
    config = cfgs['preprocess']
    generate_lab(config)
    print('Complete!')