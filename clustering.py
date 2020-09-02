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

from utils import get_mel_from_audio

sys.path.append('WaveGlow/')


def load_mels(base_path, from_audio):
    full_mels, classes = [], []
    min_len = float('inf')
    max_val = 0
    for path in os.listdir(base_path):
        if from_audio and '.wav' in path:
            mel = get_mel_from_audio(base_path + path)
        elif '.npy' in path:
            mel = np.load(base_path + path, allow_pickle=True)
            if '-' in path:
                classes.append(int(path.split('-')[0]))
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
    if len(classes) > 0:
        assert len(classes) == len(full_mels)

    return mels, max_val, classes


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, required=True, help='Path to elements for classification')
    parser.add_argument('--check_clusterizations', action='store_true', help='Check if the clusterizations is good')
    parser.add_argument('--classes_items', type=int, default=20, help='Number of clusters to create')
    parser.add_argument('--save_path', type=str, help='Path to save the outcome. Defaults to --path')
    parser.add_argument('--clusters', type=int, default=6, help='Number of clusters to create')
    parser.add_argument('--n_mel_channels', type=int, default=80, help='Number of mel channels')
    parser.add_argument('--audio', action='store_true', help='Extract mel spectrogram from audio')
    parser.add_argument('-w', '--waveglow', type=str, help='Used to generate audio of the centroids')

    args = parser.parse_args()
    n_mel_channels = args.n_mel_channels

    if args.check_clusterizations:
        """
        In order to check that the clusterizations works, the input files must have been created with
        every class the same number of items. Since KMeans will assign a class number that could not
        correspond to the original number, we have to check all possible combinations.
        """
        import itertools

        mel_spectrogram_list, _, classes = load_mels(args.path, args.audio)
        unique_classes = np.unique(np.array(classes))
        kmeans = KMeans(n_clusters=len(unique_classes), random_state=0, n_init=30).fit(mel_spectrogram_list)
        y = kmeans.labels_
        best_acc = 0
        best_classes = None
        for permutated_classes in itertools.permutations(unique_classes):
            new_classes = [c for c in permutated_classes for i in range(args.classes_items)]
            acc = sum(y == new_classes) / len(y)
            if acc > best_acc:
                best_acc = acc
                best_classes = permutated_classes

        print(f'The accuracy of the classifier is {100 * best_acc:.2f} %, with classes {best_classes}')

    else:
        save_path = args.path
        if args.save_path:
            save_path = args.save_path

        n_mel_channels = args.n_mel_channels

        mel_spectrogram_list, max_val, _ = load_mels(args.path, args.audio)
        print('All mel spectrograms were loaded. Execute K-means.')
        kmeans = KMeans(n_clusters=args.clusters, random_state=0, n_init=20).fit(mel_spectrogram_list)
        y = kmeans.labels_
        print(
            f'K-means algoritm finished executing. {"Generate centroids wav files" if args.waveglow else "Start T-SNE"}')

        if args.waveglow:
            waveglow = torch.load(args.waveglow)['model']
            waveglow.cuda().eval().half()
            for k in waveglow.convinv:
                k.float()
            progress_bar = tqdm(enumerate(kmeans.cluster_centers_), total=len(kmeans.cluster_centers_))
            progress_bar.set_description('Generating centroids wav files')
            for i, centroid in progress_bar:
                mel = centroid.reshape(n_mel_channels, -1)
                with torch.no_grad():
                    audio = waveglow.infer(torch.FloatTensor(mel * max_val).unsqueeze(0).cuda().half(), sigma=0.666)

                sf.write(f'{save_path}/centroid_{i + 1}-of-{args.clusters}.wav',
                         audio[0].to(torch.float32).data.cpu().numpy(), 22050)
            print('Finished generating. Start T-SNE')

        tsne = TSNE()
        X_embedded = tsne.fit_transform(mel_spectrogram_list)
        palette = sns.color_palette("bright", args.clusters)

        sns.scatterplot(X_embedded[:, 0], X_embedded[:, 1], hue=y, legend='full', palette=palette)
        plt.savefig(save_path + 'tsne.jpg', dpi=300)
        plt.show()
        print('T-SNE algorithm finished')
