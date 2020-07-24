import numpy as np
import pandas as pd
import torch
from librosa import load

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


def load_wav_to_torch(full_path, sampling_rate):
    data, _ = load(full_path, sampling_rate)
    return torch.FloatTensor(data.astype(np.float32))


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


def load_vesus(vesus_path):
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
