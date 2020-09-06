import argparse
import os
from random import sample

import librosa
import numpy as np
import torch

from classifier import Classifier

id_to_emotion = {
    0: 'Neutral',
    1: 'Angry',
    2: 'Happy',
    3: 'Sad',
    4: 'Fearful'
}
from_ids_savee = {
    'a': 'Angry',
    'f': 'Fearful',
    'h': 'Happy',
    'n': 'Neutral',
    'sa': 'Sad'
}
from_ids_cremad = {
    "NEU": "Neutral",
    "ANG": "Angry",
    "HAP": "Happy",
    "SAD": "Sad",
    "FEA": "Fearful"
}


def inference_folder(model, folder, dataset):
    files, correct = 0, 0
    for path in sample(os.listdir(folder), 500):
        if '.wav' not in path:
            continue
        if dataset == 'SAVEE':
            if path[0] not in from_ids_savee or path[:1] not in from_ids_savee:
                continue
            files += 1
            gt_emotion = path[0]
            if path[:1] == 'sa':
                gt_emotion = path[:1]
            gt_emotion = from_ids_savee[gt_emotion]
        elif dataset == 'CREMA-D':
            if path[9:12] not in from_ids_cremad:
                continue
            gt_emotion = from_ids_cremad[path[9:12]]
        else:
            raise ValueError('Dataset not supported')

        files += 1

        inference, p_emotion = inference_from_path(model, folder + path)
        inference_str = [str(float(int(100 * i) / 100)) for i in np.mean(inference, axis=0)]
        print(f"Inferred emotion for {path} is: {p_emotion} -> {', '.join(inference_str)} ")
        if p_emotion == gt_emotion:
            correct += 1
    print(f'Achieved accuracy of {100 * correct / files:.2f}%')


def inference_from_path(model, path):
    melspec = librosa.power_to_db(
        librosa.feature.melspectrogram(librosa.load(path)[0],
                                       sr=args.sr, n_fft=hparams.n_ftt, n_mels=hparams.n_mel_channels,
                                       hop_length=hparams.hop_length),
        ref=np.max)
    melspec = melspec / 80 + 1

    if melspec.shape[1] < hparams.n_frames:
        melspec2 = torch.zeros(80, hparams.n_frames)
        melspec2[:, :melspec.shape[1]] = torch.FloatTensor(melspec)
        melspec = melspec2

    inference = model.inference(torch.FloatTensor(melspec).cuda().half().unsqueeze(0)).cpu().detach().numpy()
    p_emotion = id_to_emotion[np.argmax(np.mean(inference, axis=0))]
    return inference, p_emotion


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--classifier_path', type=str, required=True, help='Classifier checkpoint path.')
    parser.add_argument('--path', type=str, required=True, help='Path to file to inference.')
    parser.add_argument('--hparams', type=str, help='Comma separated name=value pairs.')
    parser.add_argument('--sr', type=int, default=22050, help='Sample rate')
    parser.add_argument('--inference_folder', action='store_true',
                        help='Inference all the files from the folder given in the path.')
    parser.add_argument('--dataset', type=str, help='Dataset to use to load the labels.')

    args = parser.parse_args()

    model_info = torch.load(args.classifier_path)
    hparams = model_info['hyper_parameters']
    if 'use_labels' not in hparams and 'use_intended_labels' in hparams:
        hparams['use_labels'] = hparams['use_intended_labels']

    model = Classifier(hparams).cuda()
    model.load_state_dict(model_info['state_dict'])
    model.eval()
    if hparams.precision == 16:
        model = model.half()
    if args.inference_folder:
        inference_folder(model, args.path, args.dataset)
    else:
        inference, p_emotion = inference_from_path(model, args.path)
        print(f"Inferred emotion for {args.path} is: {p_emotion}")
