import os

import numpy as np
import pandas as pd
import torch
from librosa import load

import layers
from hparams import HParams

emo_id_to_text = {
    0: 'Neutral',
    1: 'Anger',
    2: 'Happiness',
    3: 'Sadness',
    4: 'Fear',
}


def get_mask_from_lengths(lengths):
    max_len = torch.max(lengths).item()
    ids = torch.arange(0, max_len, out=torch.cuda.LongTensor(max_len))
    mask = (ids < lengths.unsqueeze(1)).bool()
    return mask


def load_wav_to_torch(full_path, sampling_rate=22050):
    data, _ = load(full_path, sampling_rate)
    if abs(data.min()) > 1 or abs(data.max()) > 1:
        data = data / max(abs(data.min()), abs(data.max()))
    return torch.FloatTensor(data.astype(np.float32))


def get_mel_from_audio(path):
    hparams = HParams()
    n_mel_channels = hparams.n_mel_channels
    stft = layers.TacotronSTFT(hparams.filter_length, hparams.hop_length, hparams.win_length,
                               n_mel_channels, hparams.sampling_rate, hparams.mel_fmin,
                               hparams.mel_fmax)
    audio = load_wav_to_torch(path)
    audio_norm = audio / hparams.max_wav_value
    audio_norm = audio_norm.unsqueeze(0)
    audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
    return stft.mel_spectrogram(audio_norm)[0].numpy()


def load_filepaths_and_text(filename, wavs_path, split="|"):
    with open(filename, encoding='utf-8') as f:
        filepaths_and_text = []
        for line in f:
            l = line.strip().split(split)
            filepaths_and_text.append([wavs_path + l[0]] + l[1:])
    return filepaths_and_text


def calculate_emotions(labeled_emotions, labeled_intensities):
    """
    Calculate the emotions that are present in each sentence taking into consideration the values given
    by the different annotators.
    Args:
        labeled_emotions (np.ndarray): Emotions labeled as list of string numbers.
        labeled_intensities (np.ndarray): Intensity of the emotions as list of integers.

    Returns:
        dict: Key is the emotion and value the intensity.
        int: Unused emotions.
    """
    emotions = []
    n_labels = len(labeled_emotions)
    for id, emotions_str in emo_id_to_text.items():
        idxs_emotion = np.where(labeled_emotions == id)[0]
        if len(idxs_emotion) > 0:
            mean_emotion_intensity = labeled_intensities[idxs_emotion].mean() * len(idxs_emotion) / (n_labels * 5)
        else:
            mean_emotion_intensity = 0
        emotions.append(mean_emotion_intensity)

    return emotions


def load_vesus(filename, wavs_path, split="|", use_intended_labels=False, use_text=True):
    speakers, emotions = [], []
    vesus_ids = {
        "Neutral": [1, 0, 0, 0, 0],
        "Angry": [0, 1, 0, 0, 0],
        "Happy": [0, 0, 1, 0, 0],
        "Sad": [0, 0, 0, 1, 0],
        "Fearful": [0, 0, 0, 0, 1]
    }
    with open(filename, encoding='utf-8') as f:
        filepaths_and_text = []
        for line in f:
            l = line.strip().split(split)
            filepath = wavs_path + l[0]
            if use_text:
                filepath = [filepath, l[1]]
            filepaths_and_text.append(filepath)
            speakers.append(int(l[2]))
            if use_intended_labels:
                emotions.append(vesus_ids[l[0].split('/')[1]])
            else:
                emotions.append([float(i) for i in l[3].split(',')])
    return filepaths_and_text, speakers, emotions


def load_vesus_full(vesus_path):
    utterances, speakers, emotions, paths = [], [], [], []

    labels = pd.read_csv(vesus_path + '/Tools/VESUS_Key.csv', header=0)
    filepaths_and_text = []
    for row in labels.itertuples():
        file_path = vesus_path + 'Audio/' + row[1]
        actor = row[2]

        labeled_emotions = np.array([int(i) for i in row[8][1:-1].split(',')])
        labeled_intensities = np.array([int(i) for i in row[9][1:-1].split(',')])

        speakers.append(actor)
        emotions.append(calculate_emotions(labeled_emotions, labeled_intensities))
        filepaths_and_text.append([file_path, row[11].capitalize()])

    return filepaths_and_text, speakers, emotions


def to_gpu(x):
    x = x.contiguous()

    if torch.cuda.is_available():
        x = x.cuda(non_blocking=True)
    return torch.autograd.Variable(x)


def mel_to_audio(base_path, waveglow_path, randomize=True, force_create=False):
    from tqdm import tqdm
    from soundfile import write
    import sys
    sys.path.append('WaveGlow/')
    dir_list = os.listdir(base_path)
    if randomize:
        from random import shuffle
        shuffle(dir_list)

    for path in tqdm(dir_list):
        if '.npy' not in path:
            continue
        full_path = f'{base_path}/{path.split(".")[0]}.wav'
        if os.path.exists(full_path):
            if not force_create:
                print(f'File {full_path} already exists. Skip.')
                continue
            else:
                print(f'File {full_path} already exists. Creating again.')

        mel = np.load(base_path + path, allow_pickle=True)
        waveglow = torch.load(waveglow_path)['model']
        waveglow.cuda().eval().half()
        for k in waveglow.convinv:
            k.float()
        with torch.no_grad():
            audio = waveglow.infer(torch.FloatTensor(mel).unsqueeze(0).cuda().half(), sigma=0.666)
            write(full_path, audio[0].to(torch.float32).data.cpu().numpy(), 22050)
