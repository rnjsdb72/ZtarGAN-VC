import os
from glob import glob
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings(action='ignore')
import librosa
import torch
from torch.nn.utils.rnn import pad_sequence

def mp3_to_mel(n_fft = 2048, n_mels = 256, hop_length = 512,
               win_length = 2048):
    error = 0
    error_lst = []
    res_lst = []
    title_lst = []
    files = sorted(glob('./download/*.mp3'))
    print('Start Converting!')
    for idx, file in enumerate(tqdm(files)):
        try:
            sig, sr = librosa.load(file)
            mel_spec = librosa.feature.melspectrogram(sig, sr=sr, n_fft=n_fft, n_mels=n_mels, 
                                                hop_length=hop_length, win_length=win_length)
            log_mel_spec = librosa.power_to_db(mel_spec)
            res_lst.append(torch.tensor(log_mel_spec))
            title_lst.append(file.split('/')[-1].split('.')[0])
        except:
            error +=1
            error_lst.append(file)
        if len(res_lst) == 5000:
            num_data = len(res_lst)
            lst = []
            for res in res_lst:
                lst += [*torch.tensor(res)]
            res = pad_sequence(lst, batch_first=True).view(num_data,256,-1)
            if not os.path.exists('./data'):
                os.makedirs('./data')
            torch.save(res, f'./data/dataset_audio{idx}.pt')
            pd.DataFrame({'title':title_lst}).to_csv(f'./data/titles{idx}.csv', index=False)
            res_lst = []
            title_lst = []
            lst = []
    print('Convert Complete!')
    print('Number of Error: ', error)
    print('Error List:')
    print(error_lst)

if __name__ == '__main__':
    mp3_to_mel()