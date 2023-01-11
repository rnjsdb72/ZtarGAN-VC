from torch.utils import data
import torch
import glob
from os.path import join, basename
import sys
import numpy as np
from speaker_encoder.encoder.model import SpeakerEncoder
from tqdm import tqdm
from speaker_encoder.encoder import audio

min_length = 256  # Since we slice 256 frames from each utterance when training.

def to_categorical(y, num_classes=None):
    """Converts a class vector (integers) to binary class matrix.
    E.g. for use with categorical_crossentropy.
    # Arguments
        y: class vector to be converted into a matrix
            (integers from 0 to num_classes).
        num_classes: total number of classes.
    # Returns
        A binary matrix representation of the input. The classes axis
        is placed last.
    From Keras np_utils
    """
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=np.float32)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical

def embed_frames_batch(frames_batch, _model):
    """
    Computes embeddings for a batch of mel spectrogram.
    
    :param frames_batch: a batch mel of spectrogram as a numpy array of float32 of shape 
    (batch_size, n_frames, n_channels)
    :return: the embeddings as a numpy array of float32 of shape (batch_size, model_embedding_size)
    """
    if _model is None:
        raise Exception("Model was not loaded. Call load_model() before inference.")
    
    frames = torch.from_numpy(frames_batch).to('cpu').float()
    embed = _model.forward(frames[0]).detach().cpu().numpy()
    return embed


def compute_partial_slices(n_samples, args,
                           min_pad_coverage=0.75, overlap=0.5):
    """
    Computes where to split an utterance waveform and its corresponding mel spectrogram to obtain 
    partial utterances of <partial_utterance_n_frames> each. Both the waveform and the mel 
    spectrogram slices are returned, so as to make each partial utterance waveform correspond to 
    its spectrogram. This function assumes that the mel spectrogram parameters used are those 
    defined in params_data.py.
    
    The returned ranges may be indexing further than the length of the waveform. It is 
    recommended that you pad the waveform with zeros up to wave_slices[-1].stop.
    
    :param n_samples: the number of samples in the waveform
    :param partial_utterance_n_frames: the number of mel spectrogram frames in each partial 
    utterance
    :param min_pad_coverage: when reaching the last partial utterance, it may or may not have 
    enough frames. If at least <min_pad_coverage> of <partial_utterance_n_frames> are present, 
    then the last partial utterance will be considered, as if we padded the audio. Otherwise, 
    it will be discarded, as if we trimmed the audio. If there aren't enough frames for 1 partial 
    utterance, this parameter is ignored so that the function always returns at least 1 slice.
    :param overlap: by how much the partial utterance should overlap. If set to 0, the partial 
    utterances are entirely disjoint. 
    :return: the waveform slices and mel spectrogram slices as lists of array slices. Index 
    respectively the waveform and the mel spectrogram with these slices to obtain the partial 
    utterances.
    """
    assert 0 <= overlap < 1
    assert 0 < min_pad_coverage <= 1
    
    partial_utterance_n_frames = args.audio.partials_n_frames
    samples_per_frame = int((args.audio.sampling_rate * args.mel.mel_window_step / 1000))
    n_frames = int(np.ceil((n_samples + 1) / samples_per_frame))
    frame_step = max(int(np.round(partial_utterance_n_frames * (1 - overlap))), 1)

    # Compute the slices
    wav_slices, mel_slices = [], []
    steps = max(1, n_frames - partial_utterance_n_frames + frame_step + 1)
    for i in range(0, steps, frame_step):
        mel_range = np.array([i, i + partial_utterance_n_frames])
        wav_range = mel_range * samples_per_frame
        mel_slices.append(slice(*mel_range))
        wav_slices.append(slice(*wav_range))
        
    # Evaluate whether extra padding is warranted or not
    last_wav_range = wav_slices[-1]
    coverage = (n_samples - last_wav_range.start) / (last_wav_range.stop - last_wav_range.start)
    if coverage < min_pad_coverage and len(mel_slices) > 1:
        mel_slices = mel_slices[:-1]
        wav_slices = wav_slices[:-1]
    
    return wav_slices, mel_slices

def embed_utterance(wav, _model, args, using_partials=True, return_partials=False, **kwargs):
    """
    Computes an embedding for a single utterance.
    
    # TODO: handle multiple wavs to benefit from batching on GPU
    :param wav: a preprocessed (see audio.py) utterance waveform as a numpy array of float32
    :param using_partials: if True, then the utterance is split in partial utterances of 
    <partial_utterance_n_frames> frames and the utterance embedding is computed from their 
    normalized average. If False, the utterance is instead computed from feeding the entire 
    spectogram to the network.
    :param return_partials: if True, the partial embeddings will also be returned along with the 
    wav slices that correspond to the partial embeddings.
    :param kwargs: additional arguments to compute_partial_splits()
    :return: the embedding as a numpy array of float32 of shape (model_embedding_size,). If 
    <return_partials> is True, the partial utterances as a numpy array of float32 of shape 
    (n_partials, model_embedding_size) and the wav partials as a list of slices will also be 
    returned. If <using_partials> is simultaneously set to False, both these values will be None 
    instead.
    """
    # Process the entire utterance if not using partials
    if not using_partials:
        #frames = audio.wav_to_mel_spectrogram(wav)
        embed = embed_frames_batch(wav[None, ...], _model)[0]
        if return_partials:
            return embed, None, None
        return embed
    
    # Compute where to split the utterance into partials and pad if necessary
    wave_slices, mel_slices = compute_partial_slices(len(wav), args.config.data, **kwargs)
    #max_wave_length = wave_slices[-1].stop
    #if max_wave_length >= len(wav):
    #    wav = np.pad(wav, (0, max_wave_length - len(wav)), "constant")
    
    # Split the utterance into partials
    #frames = audio.wav_to_mel_spectrogram(wav)
    frames_batch = np.array([wav[s] for s in mel_slices])
    partial_embeds = embed_frames_batch(frames_batch, _model)
    
    # Compute the utterance embedding from the partial embeddings
    raw_embed = np.mean(partial_embeds, axis=0)
    embed = raw_embed / np.linalg.norm(raw_embed, 2)
    
    if return_partials:
        return embed, partial_embeds, wave_slices
    return embed

def to_embedding(y, cfg_speaker_encoder):
    """Converts a Mel Spectrogram to Embedding Vector using Speaker Encoder.
    E.g. for use with categorical_crossentropy.
    # Arguments
        y: Mel Spectrogram to be converted into a Embedding Vector
            (integers from 0 to num_classes).
        num_classes: total number of classes.
    # Returns
        A Embedding Vector of the input.
    From Keras np_utils
    """
    device = torch.device('cpu')
    loss_device = torch.device('cpu')
    # 모델 불러오기
    enc = SpeakerEncoder(device, loss_device, cfg_speaker_encoder.config)
    
    # ckpt 입력
    enc.load_state_dict(torch.load(cfg_speaker_encoder.ckpt_path, map_location=device)['model_state'], strict=False)

    # embedding 뽑기
    y_ = y.copy()
    y = np.array([y])
    embed = embed_utterance(y, enc, cfg_speaker_encoder)

    return embed


class MyDataset(data.Dataset):
    """Dataset for Mel Spectrogram features and speaker labels."""

    def __init__(self, speakers_using, cfg_speaker_encoder, data_dir, prefix):
        self.prefix = prefix
        self.speakers = speakers_using
        self.spk2idx = dict(zip(self.speakers, range(len(self.speakers))))
        self.prefix_length = len(self.speakers[0])
        self.cfg_speaker_encoder = cfg_speaker_encoder

        mc_files = glob.glob(join(data_dir, '*.npy'))
        if not self.prefix:
            mc_files = [i for i in tqdm(mc_files) if basename(i)[:self.prefix_length] in self.speakers]
        else:
            mc_files = mc_files = [i for i in tqdm(mc_files) if basename(i)[self.prefix[0]:self.prefix[1]] in self.speakers]
        self.mc_files = self.rm_too_short_utt(mc_files)
        self.num_files = len(self.mc_files)
        print("\t Number of training samples: ", self.num_files)
        for f in tqdm(self.mc_files):
            mc = np.load(f).T
            if mc.shape[0] <= min_length:
                print(f)
                raise RuntimeError(
                    f"The data may be corrupted! We need all MCEP features having more than {min_length} frames!")

    def rm_too_short_utt(self, mc_files, min_length=min_length):
        new_mc_files = []
        for mc_file in tqdm(mc_files):
            mc = np.load(mc_file).T
            if mc.shape[0] > min_length:
                new_mc_files.append(mc_file)
        return new_mc_files

    def sample_seg(self, feat, sample_len=min_length):
        assert feat.shape[0] - sample_len >= 0
        s = np.random.randint(0, feat.shape[0] - sample_len + 1)
        return feat[s:s + sample_len, :]

    def __len__(self):
        return self.num_files

    def __getitem__(self, index):
        filename = self.mc_files[index]
        spk = basename(filename).split('_')[0]
        spk_idx = self.spk2idx[spk]
        mc_ = np.load(filename).T
        mc = mc_.copy()
        mc = self.sample_seg(mc)
        mc = np.transpose(mc, (1, 0))  # (T, D) -> (D, T), since pytorch need feature having shape
        # to one-hot
        spk_emb = np.squeeze(to_embedding(mc_, self.cfg_speaker_encoder, num_classes=len(self.speakers)))
        spk_cat = np.squeeze(to_categorical([spk_idx], num_classes=len(self.speakers)))
        
        return torch.FloatTensor(mc), torch.LongTensor([spk_idx]).squeeze_(), torch.FloatTensor(spk_emb), torch.FloatTensor(spk_cat)


class TestDataset(object):
    """Dataset for testing."""

    def __init__(self, speakers_using, cfg_speaker_encoder, data_dir, wav_dir, src_spk='p262', trg_spk='p272'):
        self.speakers = speakers_using
        self.spk2idx = dict(zip(self.speakers, range(len(self.speakers))))
        self.prefix_length = len(self.speakers[0])

        self.src_spk = src_spk
        self.trg_spk = trg_spk
        self.mc_files = sorted(glob.glob(join(data_dir, '{}*.npy'.format(self.src_spk))))
        
        self.src_wav_dir = f'{wav_dir}/{src_spk}'
        self.trg_wav_dir = f'{wav_dir}/{trg_spk}'
        self.spk_idx_src, self.spk_idx_trg = self.spk2idx[src_spk.replace('*', '')], self.spk2idx[trg_spk.replace('*', '')]
        
        try:
            self.src_mc = np.load(self.src_wav_dir).T
            self.trg_mc = np.load(self.trg_wav_dir).T
        except:
            self.src_mc = np.load(glob.glob(self.src_wav_dir+'*')[0]).T
            self.trg_mc = np.load(glob.glob(self.trg_wav_dir+'*')[0]).T

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


def get_loader(speakers_using, cfg_speaker_encoder, data_dir, batch_size=32, mode='train', num_workers=1, prefix=None):
    dataset = MyDataset(speakers_using, cfg_speaker_encoder, data_dir, prefix)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=(mode == 'train'),
                                  num_workers=num_workers,
                                  drop_last=True)
    return data_loader
