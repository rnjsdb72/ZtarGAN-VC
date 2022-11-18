import sys
import json
import warnings
from collections import namedtuple
import wave
from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from utils import *
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import glob
from os.path import join, basename
import subprocess
import torch
import os
import numpy as np
import librosa

import audio as Audio

warnings.filterwarnings("ignore")


def resample(spk_folder, sampling_rate, origin_wavpath, target_wavpath):
    """
    Resample files to x frames and save to output dir.
    :param spk_folder: speaker dir
    :param sampling_rate: frame rate to resample to
    :param origin_wavpath: root path of all speaker folders to resample
    :param target_wavpath: root path of resampled speakers to output to
    :return: None
    """
    wavfiles = [i for i in os.listdir(join(origin_wavpath, spk_folder)) if i.endswith('.wav')]
    for wav in wavfiles:
        folder_to = join(target_wavpath, spk_folder)
        os.makedirs(folder_to, exist_ok=True)
        wav_to = join(folder_to, wav)
        wav_from = join(origin_wavpath, spk_folder, wav)
        subprocess.call(['sox', wav_from, '-r', str(sampling_rate), wav_to])

    return None


def resample_to_xk(sampling_rate, origin_wavpath, target_wavpath, num_workers=1):
    """
    Prepare folders for resmapling at x frames.
    :param sampling_rate: frame rate to resample to
    :param origin_wavpath: root path of all speaker folders to resample
    :param target_wavpath: root path of resampled speakers to output to
    :param num_workers: cpu workers
    :return: None
    """
    os.makedirs(target_wavpath, exist_ok=True)
    spk_folders = os.listdir(origin_wavpath)
    print(f'> Using {num_workers} workers!')
    executor = ProcessPoolExecutor(max_workers=num_workers)

    futures = []
    for spk_folder in tqdm(spk_folders):
        futures.append(executor.submit(partial(resample, spk_folder, sampling_rate, origin_wavpath, target_wavpath)))

    result_list = [future.result() for future in tqdm(futures)]
    print('Completed!')

    return None

def split_data(paths):
    """
    Split path data into train test split.
    :param paths: all wav paths of a speaker dir.
    :return: train wav paths, test wav paths
    """
    indices = np.arange(len(paths))
    test_size = 0.1
    train_indices, test_indices = train_test_split(indices, test_size=test_size, random_state=1234)
    train_paths = list(np.array(paths)[train_indices])
    test_paths = list(np.array(paths)[test_indices])

    return train_paths, test_paths

def get_mel_from_wav(audio, _stft, sample_rate):
    audio, _ = librosa.load(audio, sr=sample_rate, mono=True)
    audio = torch.clip(torch.FloatTensor(audio).unsqueeze(0), -1, 1)
    audio = torch.autograd.Variable(audio, requires_grad=False)
    melspec, energy = _stft.mel_spectrogram(audio)
    melspec = torch.squeeze(melspec, 0).numpy().astype(np.float32)
    energy = torch.squeeze(energy, 0).numpy().astype(np.float32)

    return melspec

def get_spk_mel_feats(spk_name, spk_paths, output_dir, sample_rate, config):
    """
    Convert wav files to there Mel features.
    :param spk_name: name of speaker dir
    :param spk_paths: paths of all wavs in speaker dir
    :param output_dir: dir to output Mels to
    :param sample_rate: frame rate of wav files
    :return: None
    """
    #f0s = []
    #coded_sps = []
    #for wav_file in spk_paths:
    #    f0, _, _, _, coded_sp = world_encode_wav(wav_file, fs=sample_rate)
    #    f0s.append(f0)
    #    coded_sps.append(coded_sp)

    #log_f0s_mean, log_f0s_std = logf0_statistics(f0s)
    #coded_sps_mean, coded_sps_std = coded_sp_statistics(coded_sps)

    #np.savez(join(output_dir, spk_name + '_stats.npz'),
    #         log_f0s_mean=log_f0s_mean,
    #         log_f0s_std=log_f0s_std,
    #         coded_sps_mean=coded_sps_mean,
    #         coded_sps_std=coded_sps_std)
    
    STFT = Audio.stft.TacotronSTFT(
            config.Mel_preprocess.stft.filter_length,
            config.Mel_preprocess.stft.hop_length,
            config.Mel_preprocess.stft.win_length,
            config.Mel_preprocess.mel.n_mel_channels,
            config.Mel_preprocess.audio.sampling_rate,
            config.Mel_preprocess.mel.mel_fmin,
            config.Mel_preprocess.mel.mel_fmax,
        )
    
    for wav_file in spk_paths:
        wav_name = basename(wav_file)
        mel_spectrogram = get_mel_from_wav(wav_file, STFT, sample_rate)
        np.save(os.path.join(output_dir, wav_name.replace('.wav', '.npy')),
                mel_spectrogram,
                allow_pickle=False)

    return None


def process_spk(spk_path, mc_dir, sample_rate, config):
    """
    Prcoess speaker wavs to Mels
    :param spk_path: path to speaker wav dir
    :param mc_dir: output dir for speaker data
    :return: None
    """
    spk_paths = glob.glob(join(spk_path, '*.wav'))

    # find the sampling rate of teh wav files you are about to convert

    spk_name = basename(spk_path)

    get_spk_mel_feats(spk_name, spk_paths, mc_dir, sample_rate, config)

    return None


def process_spk_with_split(spk_path, mc_dir_train, mc_dir_test, sample_rate, config):
    """
    Perform train test split on a speaker and process wavs to Mels.
    :param spk_path: path to speaker wav dir
    :param mc_dir_train: output dir for speaker train data
    :param mc_dir_test: output dir for speaker test data
    :return: None
    """
    spk_paths = glob.glob(join(spk_path, '*.wav'))

    if len(spk_paths) == 0:
        return None
    
    # find the samplng rate of the wav files you are about to convert

    spk_name = basename(spk_path)
    train_paths, test_paths = split_data(spk_paths)

    get_spk_mel_feats(spk_name, train_paths, mc_dir_train, sample_rate, config)
    get_spk_mel_feats(spk_name, test_paths, mc_dir_test, sample_rate, config)

    return None


if __name__ == '__main__':
    args = arg_parse()
    with open(args.cfgs, 'r') as f:
        cfgs = json.load(f, object_hook=lambda d: namedtuple('x', d.keys())(*d.values()))
        
    # DATA SPLITTING.
    perform_data_split = cfgs.data_split.perform_data_split
    resample_rate = cfgs.resample_rate
    origin_wavpath = cfgs.data_split.origin_wavpath
    target_wavpath = cfgs.data_split.target_wavpath
    origin_wavpath_train = cfgs.data_split.origin_wavpath_train
    origin_wavpath_eval = cfgs.data_split.origin_wavpath_eval
    target_wavpath_train = cfgs.data_split.target_wavpath_train
    target_wavpath_eval = cfgs.data_split.target_wavpath_eval
    mc_dir_train = cfgs.Mel_preprocess.mel_dir_train
    mc_dir_test = cfgs.Mel_preprocess.mel_dir_test
    sample_rate = cfgs.Mel_preprocess.audio.sampling_rate

    # Do resample.
    if perform_data_split == 'n':
        if resample_rate > 0:
            print(f'Resampling speakers in {origin_wavpath_train} to {target_wavpath_train} at {resample_rate}')
            resample_to_xk(resample_rate, origin_wavpath_train, target_wavpath_train)
            print(f'Resampling speakers in {origin_wavpath_eval} to {target_wavpath_eval} at {resample_rate}')
            resample_to_xk(resample_rate, origin_wavpath_eval, target_wavpath_eval)
    else:
        if resample_rate > 0:
            print(f'Resampling speakers in {origin_wavpath} to {target_wavpath} at {resample_rate}')
            resample_to_xk(resample_rate, origin_wavpath, target_wavpath)

    print('Making directories for Mels...')
    os.makedirs(mc_dir_train, exist_ok=True)
    os.makedirs(mc_dir_test, exist_ok=True)

    speakers = os.listdir(origin_wavpath)
    if perform_data_split == 'n':
        # current wavs working with (train)
        working_train_dir = target_wavpath_train
        for spk in tqdm(speakers):
            spk_dir = os.path.join(working_train_dir, spk)
            process_spk(spk_dir, mc_dir_train, sample_rate, cfgs)

        # current wavs working with (eval)
        working_eval_dir = target_wavpath_eval
        for spk in tqdm(speakers):
            spk_dir = os.path.join(working_eval_dir, spk)
            process_spk(spk_dir, mc_dir_test, sample_rate, cfgs)
    else:
        # current wavs we are working with (all for data split)
        working_dir = target_wavpath
        outer_bar = tqdm(speakers, position=0)
        for spk in outer_bar:
            spk_dir = os.path.join(working_dir, spk)
            process_spk_with_split(spk_dir, mc_dir_train, mc_dir_test, sample_rate, cfgs)
    
    print('Completed!')

    sys.exit(0)