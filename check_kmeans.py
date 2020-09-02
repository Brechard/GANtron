import argparse
import itertools
import os

import numpy as np
from sklearn.cluster import KMeans
from tqdm import tqdm

from utils import get_mel_from_audio


def load_mels(base_path):
    full_mels, classes = [], []
    min_len = float('inf')
    max_val = 0
    n_mel_channels = 80
    emotions = []
    em_id = 0
    npys = []
    for emotion in tqdm(os.listdir(base_path)):
        if '.' in emotion:
            continue
        for path in os.listdir(base_path + emotion):
            if '.wav' not in path and '.npy' not in path:
                continue
            if '.npy' in path:
                mel = np.load(base_path + emotion + '/' + path, allow_pickle=True)
                npys.append(path.split('.')[0])
            elif path.split('.')[0] not in npys:
                mel = get_mel_from_audio(base_path + emotion + '/' + path)
                np.save(base_path + emotion + '/' + path.split('.')[0] + '.npy', mel)
            else:
                continue

            if mel.shape[1] < min_len:
                min_len = mel.shape[1]
            if abs(mel.min()) > max_val:
                max_val = abs(mel.min())
            if abs(mel.max()) > max_val:
                max_val = abs(mel.max())

            full_mels.append(mel)
            emotions.append(em_id)
        em_id += 1

    mels = np.zeros((len(full_mels), n_mel_channels * min_len))
    for i, mel in enumerate(full_mels):
        mels[i] = mel[:, :min_len].flatten() / max_val

    return mels, np.array(emotions)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio_path', type=str, help='Path to audio files')

    args = parser.parse_args()

    mel_spectrogram_list, emotions = load_mels(args.audio_path)
    kmeans = KMeans(n_clusters=5, random_state=0, n_init=30).fit(mel_spectrogram_list)
    y = kmeans.labels_
    acc = sum(y == emotions) / len(y)
    print(f'Basic accuracy is {100 * acc:.2f} %')

    unique_emotions = np.unique(emotions)
    best_acc = 0
    best_classes = None
    for permutated_classes in itertools.permutations(unique_emotions):
        new_classes = [c for c in permutated_classes for i in range(252)]
        acc = sum(y == new_classes) / len(y)
        if acc > best_acc:
            best_acc = acc
            best_classes = permutated_classes

    print(f'The accuracy of the classifier is {100 * best_acc:.2f} %, with classes {permutated_classes}')
