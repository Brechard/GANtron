import argparse

import librosa
import numpy as np
import torch

from classifier import Classifier
from hparams_classifier import HParams

id_to_emotion = {
    0: 'Neutral',
    1: 'Angry',
    2: 'Happy',
    3: 'Sad',
    4: 'Fearful'
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--classifier_path', type=str, required=True, help='Classifier checkpoint path')
    parser.add_argument('--path', type=str, required=True, help='Path to file to inference')
    parser.add_argument('--hparams', type=str, help='Comma separated name=value pairs')
    parser.add_argument('--sr', type=int, default=22050, help='Sample rate')

    args = parser.parse_args()
    hparams_classifier = HParams(args.hparams)
    hparams_classifier.add_params(args)

    model = Classifier(hparams_classifier.__dict__).cuda()
    model.load_state_dict(torch.load(args.classifier_path)['state_dict'])
    model.eval()
    if hparams_classifier.precision == 16:
        model = model.half()

    melspec = librosa.power_to_db(
        librosa.feature.melspectrogram(librosa.load(args.path)[0],
                                       sr=args.sr, n_fft=1024, n_mels=80, hop_length=256),
        ref=np.max) / 80 + 1
    melspec2 = melspec[:, hparams_classifier.mel_offset:hparams_classifier.mel_offset + hparams_classifier.n_frames]
    inference = model.inference(torch.FloatTensor(melspec2).cuda().half().unsqueeze(0))
    print(f"Inferred emotions is: {id_to_emotion[np.argmax(inference.cpu().detach().numpy())]}")
