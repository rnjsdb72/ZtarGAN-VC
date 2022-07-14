import os
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as AT
from torch.nn.utils.rnn import pad_sequence

class mp3_to_mel():
    def __init__(self, sr:int = 44100, n_fft:int = 2048, n_mels:int = 128,
                 hop_length:int = 512, win_length:int = 2048):
        self.n_mels = n_mels
        self.to_mel = nn.Sequential(
                                AT.MelSpectrogram(
                                    sample_rate = sr,
                                    n_fft = n_fft,
                                    win_length = win_length,
                                    hop_length = hop_length,
                                    n_mels = n_mels
                                ),
                                AT.AmplitudeToDB()
                            )
        
    def load_signals(self, files):
        c = torchaudio.load(files[0])[0].shape[0]
        
        print('Loading Audio...')
        sigs = list(map(lambda x: torchaudio.load(x)[0], files))
        
        lst = []
        for sig in tqdm(sigs):
            lst += [*sig]
        
        print('Start to padding...')
        self.sig_padded = pad_sequence(lst, batch_first=True).view(len(files), 2, -1)
        
        print('Complete!')
        
    def convert_to_mel(self):
        mels = self.to_mel(self.sig_padded)
        return mels

if __name__ == '__main__':
    files = glob('./download/*.mp3')
    converter = mp3_to_mel()
    converter.load_signals(files)
    mels = converter.convert_to_mel()
    torch.save(mels, './data/data_audio.pt')
    print('Save Success!')

