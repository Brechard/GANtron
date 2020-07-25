import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import soundfile as sf
import torch
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from tqdm import tqdm

import layers
from hparams import HParams
from utils import load_wav_to_torch

sys.path.append('WaveGlow/')


# base_path = 'C:/Users/rodri/PycharmProjects/GANtron/test1/'


def get_mel_from_audio(path):
    audio, sampling_rate = load_wav_to_torch(path)
    audio_norm = audio / hparams.max_wav_value
    audio_norm = audio_norm.unsqueeze(0)
    audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
    return stft.mel_spectrogram(audio_norm)[0].numpy()


def load_mels(base_path, from_audio):
    full_mels = []
    min_len = float('inf')
    max_val = 0
    for path in os.listdir(base_path):
        if from_audio and '.wav' in path:
            mel = get_mel_from_audio(base_path + path)
        elif '.npy' in path:
            mel = np.load(base_path + path, allow_pickle=True)
        else:
            continue
        if mel.shape[1] < min_len:
            min_len = mel.shape[1]
        if abs(mel.min()) > max_val:
            max_val = abs(mel.min())
        if abs(mel.max()) > max_val:
            max_val = abs(mel.max())

        full_mels.append(mel)

    mels = np.zeros((len(full_mels), n_mel_channels * min_len))
    for i, mel in enumerate(full_mels):
        if mel.shape[0] == 1:
            mel = mel[0]
        mels[i] = mel[:, :min_len].flatten() / max_val
        # mels[i, -1] = mel.shape[1]

    return mels, max_val


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, required=True, help='Path to elements for classification')
    parser.add_argument('--save_path', type=str, help='Path to save the outcome. Defaults to --path')
    parser.add_argument('--clusters', type=int, default=3, help='Number of clusters to create')
    parser.add_argument('--n_mel_channels', type=int, default=80, help='Number of mel channels')
    parser.add_argument('--audio', action='store_true', help='Extract mel spectrogram from audio')
    parser.add_argument('-w', '--waveglow', type=str, help='Used to generate audio of the centroids')

    args = parser.parse_args()

    save_path = args.path
    if args.save_path:
        save_path = args.save_path

    n_mel_channels = args.n_mel_channels
    if args.audio:
        hparams = HParams()
        n_mel_channels = hparams.n_mel_channels
        stft = layers.TacotronSTFT(hparams.filter_length, hparams.hop_length, hparams.win_length,
                                   n_mel_channels, hparams.sampling_rate, hparams.mel_fmin,
                                   hparams.mel_fmax)

    mel_spectrogram_list, max_val = load_mels(args.path, args.audio)
    print('All mel spectrograms were loaded. Execute K-means.')
    kmeans = KMeans(n_clusters=args.clusters, random_state=0, n_init=20).fit(mel_spectrogram_list)
    y = kmeans.labels_
    print(f'K-means algoritm finished executing. {"Generate centroids wav files" if args.waveglow else "Start T-SNE"}')

    if args.waveglow:
        waveglow = torch.load(args.waveglow)['model']
        waveglow.cuda().eval().half()
        for k in waveglow.convinv:
            k.float()
        progress_bar = tqdm(enumerate(kmeans.cluster_centers_), total=args.clusters)
        progress_bar.set_description('Generating centroids wav files')
        for i, centroid in progress_bar:
            mel = centroid.reshape(n_mel_channels, -1)
            with torch.no_grad():
                audio = waveglow.infer(torch.FloatTensor(mel * max_val).unsqueeze(0).cuda().half(), sigma=0.666)

            sf.write(f'{save_path}/centroid_{i + 1}-of-{args.clusters}.wav', audio[0].to(torch.float32).data.cpu().numpy(), 22050)
        print('Finished generating. Start T-SNE')

    tsne = TSNE()
    X_embedded = tsne.fit_transform(mel_spectrogram_list)
    palette = sns.color_palette("bright", args.clusters)

    sns.scatterplot(X_embedded[:, 0], X_embedded[:, 1], hue=y, legend='full', palette=palette)
    plt.savefig(save_path + 'tsne.jpg', dpi=300)
    plt.show()
    print('T-SNE algorithm finished')
