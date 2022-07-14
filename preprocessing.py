import os
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt

import argparse

import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as AT
from torch.nn.utils.rnn import pad_sequence

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=-1)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--sr', type=int, default=44100)
    parser.add_argument('--n_fft', type=int, default=2048)
    parser.add_argument('--n_mels', type=int, default=128)
    parser.add_argument('--hop_length', type=int, default=512)
    parser.add_argument('--win_length', type=int, default=2048)
    args = parser.parse_args()
    return args

class MP3_to_Dataset(torch.utils.data.Dataset):
    def __init__(self, files, sr:int = 44100, n_fft:int = 2048, n_mels:int = 128,
                 hop_length:int = 512, win_length:int = 2048):
        super().__init__()
        self.files = files
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
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        sig = torchaudio.load(files[idx])[0]
        if sig.shape[1] < 15876000:
            sig = pad_sequence([*sig, torch.ones(15876000)], batch_first=True)[:2]
        else:
            sig = sig[:,:15876000]
        return sig
    
class Sig_to_Mel(nn.Module):
    def __init__(self, sr:int = 44100, n_fft:int = 2048, n_mels:int = 128,
                 hop_length:int = 512, win_length:int = 2048):
        super().__init__()
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
    def convert(self, sig):
        mels = self.to_mel(sig)
        return mels

def convert(files, start:int = 0, end:int = -1, batch_size:int = 64,
            sr:int = 44100, n_fft:int = 2048, n_mels:int = 128,
            hop_length:int = 512, win_length:int = 2048):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    dataset = MP3_to_Dataset(files, sr, n_fft, n_mels, hop_length, win_length)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    
    converter = Sig_to_Mel(sr, n_fft, n_mels, hop_length, win_length).to(device)
    
    if not os.path.exists('./data'):
        os.makedirs('./data')
        
    for idx, data in enumerate(tqdm(dataloader)):
        mel_spec = converter.convert(data)
        torch.save(mel_spec, f'./data/mel_spec_batch{start}_{end-1}_{idx}.pt')

if __name__ == '__main__':
    args = arg_parse()
    start = args.start
    end = args.end
    batch_size = args.batch_size
    sr = args.sr
    n_fft = args.n_fft
    n_mels = args.n_mels
    hop_length = args.hop_length
    win_length = args.win_length
    
    files = glob('./download/*.mp3')
    if end == -1:
        end = len(files)
    else:
        end += 1
    files = files[start:end]
    
    convert(files, start, end, batch_size, sr, n_fft, n_mels, hop_length, win_length)
    print('Save Succeess!')

