import re
from string import punctuation
import json

import torch
import numpy as np
from torch.utils.data import DataLoader
from g2p_en import G2p
from pypinyin import pinyin, Style

from utils.model import get_model, get_vocoder
from utils.tools import to_device, synth_samples, arg_parse
from dataset import TextDataset
from text import text_to_sequence
from text.korean import tokenize, normalize_nonchar

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def read_lexicon(lex_path):
    lexicon = {}
    with open(lex_path) as f:
        for line in f:
            temp = re.split(r"\s+", line.strip("\n"))
            word = temp[0]
            phones = temp[1:]
            if word.lower() not in lexicon:
                lexicon[word.lower()] = phones
    return lexicon


def preprocess_english(text, preprocess_config):
    text = text.rstrip(punctuation)
    lexicon = read_lexicon(preprocess_config["path"]["lexicon_path"])

    g2p = G2p()
    phones = []
    words = re.split(r"([,;.\-\?\!\s+])", text)
    for w in words:
        if w.lower() in lexicon:
            phones += lexicon[w.lower()]
        else:
            phones += list(filter(lambda p: p != " ", g2p(w)))
    phones = "{" + "}{".join(phones) + "}"
    phones = re.sub(r"\{[^\w\s]?\}", "{sp}", phones)
    phones = phones.replace("}{", " ")

    print("Raw Text Sequence: {}".format(text))
    print("Phoneme Sequence: {}".format(phones))
    sequence = np.array(
        text_to_sequence(
            phones, preprocess_config["preprocessing"]["text"]["text_cleaners"]
        )
    )

    return np.array(sequence)


def preprocess_korean(text, preprocess_config):
    lexicon = read_lexicon(preprocess_config["path"]["lexicon_path"])

    phones = []
    words = filter(None, re.split(r"([,;.\-\?\!\s+])", text))
    for w in words:
        if w in lexicon:
            phones += lexicon[w]
        else:
            phones += list(filter(lambda p: p != " ", tokenize(w, norm=False)))
    phones = "{" + "}{".join(phones) + "}"
    phones = normalize_nonchar(phones, inference=True)
    phones = phones.replace("}{", " ")

    print("Raw Text Sequence: {}".format(text))
    print("Phoneme Sequence: {}".format(phones))
    sequence = np.array(
        text_to_sequence(
            phones, preprocess_config["preprocessing"]["text"]["text_cleaners"]
        )
    )

    return np.array(sequence)


def synthesize(model, step, configs, vocoder, batchs, control_values):
    preprocess_config, model_config, train_config = configs
    pitch_control, energy_control, duration_control = control_values

    for batch in batchs:
        batch = to_device(batch, device)
        with torch.no_grad():
            # Forward
            output = model(
                *(batch[2:]),
                p_control=pitch_control,
                e_control=energy_control,
                d_control=duration_control
            )
            synth_samples(
                batch,
                output,
                vocoder,
                model_config,
                preprocess_config,
                train_config["path"]["result_path"],
            )


if __name__ == "__main__":
    args = arg_parse()
    with open(args.cfg, 'r') as f:
        cfgs = json.load(f)

    # Check source texts
    if cfgs['mode'] == "batch":
        assert cfgs['source'] is not None and cfgs['text'] is None
    if cfgs['mode'] == "single":
        assert cfgs['source'] is None and cfgs['text'] is not None

    # Read Config
    preprocess_config = cfgs['preprocess']
    model_config = cfgs['model']
    train_config = cfgs['train']
    configs = (preprocess_config, model_config, train_config)

    # Get model
    model = get_model(cfgs, configs, device, train=False)

    # Load vocoder
    vocoder = get_vocoder(model_config, device)

    # Preprocess texts
    if cfgs['mode'] == "batch":
        # Get dataset
        dataset = TextDataset(cfgs['source'], preprocess_config)
        batchs = DataLoader(
            dataset,
            batch_size=8,
            collate_fn=dataset.collate_fn,
        )
    if cfgs['mode'] == "single":
        ids = raw_texts = [cfgs['text'][:100]]
        speakers = np.array([cfgs['speaker_id']])
        if preprocess_config["preprocessing"]["text"]["language"] == "kr":
            texts = np.array([preprocess_korean(cfgs['text'], preprocess_config)])
        elif preprocess_config["preprocessing"]["text"]["language"] == "en":
            texts = np.array([preprocess_english(cfgs['text'], preprocess_config)])
        text_lens = np.array([len(texts[0])])
        batchs = [(ids, raw_texts, speakers, texts, text_lens, max(text_lens))]

    control_values = cfgs['pitch_control'], cfgs['energy_control'], cfgs['duration_control']

    synthesize(model, cfgs['restore_step'], configs, vocoder, batchs, control_values)