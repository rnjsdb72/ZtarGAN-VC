import os
from glob import glob
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings(action='ignore')
import librosa

def mp3_to_mel(n_fft = 2048, n_mels = 256, hop_length = 512, win_length = 2048):
    error = 0
    error_lst = []
    files = sorted(glob('./download/*.mp3'))
    print('Start Converting!')
    if not os.path.exists('./data'):
        os.makedirs('./data')
    for file in tqdm(files):
        try:
            sig, sr = librosa.load(file)
            mel_spec = librosa.feature.melspectrogram(sig, sr=sr, n_fft=n_fft, n_mels=n_mels, 
                                                hop_length=hop_length, win_length=win_length)
            log_mel_spec = librosa.power_to_db(mel_spec)
            np.save(f"./data/{file.split('/')[-1].split('.')[0]}.npy", log_mel_spec)
        except:
            error +=1
            error_lst.append(file)
    print('Convert Complete!')
    print('Number of Error: ', error)
    print('Error List:')
    print(error_lst)

if __name__ == '__main__':
    mp3_to_mel()

