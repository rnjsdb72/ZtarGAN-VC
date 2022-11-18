import argparse
import json
from collections import namedtuple
from model import Generator
from torch.autograd import Variable
import torch
import torch.nn.functional as F
import numpy as np
import os
from os.path import join, basename, dirname, split
import time
import datetime
from data_loader import to_categorical, to_embedding
import librosa
from utils import *
from glob import glob

import audio as Audio
from preprocess import get_mel_from_wav

class TestDataset(object):
    """Dataset for testing."""

    def __init__(self, speakers_using, config):
        self.speakers = speakers_using
        self.spk2idx = dict(zip(self.speakers, range(len(self.speakers))))
        self.prefix_length = len(self.speakers[0])

        data_dir = config.directories.test_data_dir
        wav_dir = config.directories.wav_dir
        src_spk = config.model.src_spk
        trg_spk = config.model.trg_spk
        self.src_spk = src_spk
        self.trg_spk = trg_spk
        self.mc_files = sorted(glob(join(data_dir, '{}*.npy'.format(self.src_spk))))
        
        self.src_wav_dir = f'{wav_dir}/{src_spk}'
        self.trg_wav_dir = f'{wav_dir}/{trg_spk}'
        self.spk_idx_src, self.spk_idx_trg = self.spk2idx[src_spk.replace('*', '')], self.spk2idx[trg_spk.replace('*', '')]
        
        try:
            self.src_mc = np.load(self.src_wav_dir).T
            self.trg_mc = np.load(self.trg_wav_dir).T
        except:
            print(glob(self.src_wav_dir + "/*.npy"))
            print(glob(self.trg_wav_dir + "/*.npy"))
            self.src_mc = np.load(glob(self.src_wav_dir + "/*.npy")[0]).T
            self.trg_mc = np.load(glob(self.trg_wav_dir + "/*.npy")[0]).T

        cfg_speaker_encoder = config.speaker_encoder
        spk_emb_src = to_embedding(self.src_mc, cfg_speaker_encoder, num_classes=len(self.speakers))
        spk_emb_trg = to_embedding(self.trg_mc, cfg_speaker_encoder, num_classes=len(self.speakers))
        spk_cat_src = to_categorical([self.spk_idx_src], num_classes=len(self.speakers))
        spk_cat_trg = to_categorical([self.spk_idx_trg], num_classes=len(self.speakers))
        self.spk_emb_src = spk_emb_src
        self.spk_emb_trg = spk_emb_trg
        self.spk_c_org = spk_cat_src
        self.spk_c_trg = spk_cat_trg

    def get_batch_test_data(self, batch_size=8):
        batch_data = []
        for i in range(batch_size):
            mc_file = self.mc_files[i]
            filename = basename(mc_file).split('-')[-1]
            wavfile_path = join(self.src_wav_dir, filename.replace('npy', 'wav'))
            batch_data.append(wavfile_path)
        return batch_data

    def get_batch_test_npy_data(self, batch_size=8):
        batch_data = []
        for i in range(batch_size):
            mc_file = self.mc_files[i]
            filename = basename(mc_file).split('-')[-1]
            wavfile_path = join(self.src_wav_dir, filename)
            batch_data.append(wavfile_path)
        return batch_data


def load_wav(wavfile, sr=16000):
    wav, _ = librosa.load(wavfile, sr=sr, mono=True)
    return wav_padding(wav, sr=sr, frame_period=5, multiple = 4)  # TODO
    # return wav

def test(speakers, config):
    os.makedirs(join(config.directories.convert_dir, str(config.model.resume_iters)), exist_ok=True)
    sampling_rate, num_mcep, frame_period=16000, 36, 5
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    G = Generator().to(device)
    test_loader = TestDataset(speakers, config)
    # Restore model
    print(f'Loading the trained models from step {config.model.resume_iters}...')
    G_path = join(config.directories.model_save_dir, f'{config.model.resume_iters}-G.ckpt')
    G.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))

    # Read a batch of testdata
    #test_wavfiles = test_loader.get_batch_test_data(batch_size=config.model.num_converted_wavs)
    test_wavfiles = test_loader.get_batch_test_npy_data(batch_size=config.model.num_converted_wavs)
    #test_wavs = [load_wav(wavfile, sampling_rate) for wavfile in test_wavfiles]
    test_wavs = test_wavfiles.copy()


    STFT = Audio.stft.TacotronSTFT(
                config.preprocessing.stft.filter_length,
                config.preprocessing.stft.hop_length,
                config.preprocessing.stft.win_length,
                config.preprocessing.mel.n_mel_channels,
                config.preprocessing.audio.sampling_rate,
                config.preprocessing.mel.mel_fmin,
                config.preprocessing.mel.mel_fmax,
            )

    with torch.no_grad():
        for idx, wav in enumerate(test_wavs):
            wav_name = basename(test_wavfiles[idx])
            mel_spectrogram = torch.FloatTensor([test_loader.src_mc]).unsqueeze(1).to(device)
            spk_conds = torch.FloatTensor([test_loader.spk_emb_trg]).to(device)
            converted = G(mel_spectrogram, spk_conds).data
            vocoder = get_vocoder(config, device)
            synth_samples(wav_name, converted, vocoder, config, config.directories.convert_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Model configuration.
    parser.add_argument('cfg')

    config = parser.parse_args()
    
    with open(config.cfg, 'r') as f:
        config = json.load(f, object_hook=lambda d: namedtuple('x', d.keys())(*d.values()))
        
    # no. of spks
    speakers = list(set(map(lambda x: x.split('/')[-1].split('_')[0], glob(config.directories.train_data_dir+'/*'))))
    
    print(config)
    if config.model.resume_iters is None:
        raise RuntimeError("Please specify the step number for resuming.")
    test(speakers, config)