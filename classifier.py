import argparse
import os
from random import shuffle

import numpy as np
import torch
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm

import layers
from hparams import HParams
from utils import load_vesus
from utils import load_wav_to_torch
import matplotlib.pyplot as plt

emotion_to_id = {
    'Neutral': 0,
    'Angry': 1,
    'Happy': 2,
    'Sad': 3,
    'Fearful': 4
}


def module(size_in, size_out):
    return torch.nn.Sequential(
        torch.nn.Linear(size_in, size_out),
        torch.nn.BatchNorm1d(size_out),
        torch.nn.Dropout(0.2),
        torch.nn.LeakyReLU(0.1)
    )


def conv_module(size_in, size_out):
    return torch.nn.Sequential(
        torch.nn.Conv1d(size_in, size_out, 5, padding=2),
        torch.nn.Dropout(0.2),
        torch.nn.LeakyReLU(0.1)
    )


class MelLoader(torch.utils.data.Dataset):
    def __init__(self, mel_paths, emotions):
        self.mel_paths = mel_paths
        self.emotions = emotions
        self.indexes = list(range(len(mel_paths)))
        shuffle(self.indexes)

    def get_mel(self, path):
        return torch.FloatTensor(np.load(path, allow_pickle=True))

    def __getitem__(self, index):
        mel = self.get_mel(self.mel_paths[self.indexes[index]])
        em = torch.FloatTensor(self.emotions[self.indexes[index]])
        # fig, ax = plt.subplots(figsize=(6, 4))
        # im = plt.imshow(mel.cpu().numpy(), origin='lower')
        # plt.title(self.mel_paths[self.indexes[index]])
        # fig.colorbar(im, ax=ax)
        # plt.show()
        # plt.imshow(librosa.feature.melspectrogram(librosa.load('C:/Users/rodri/Datasets/VESUS/Audio/1/Happy/100.wav')[0], sr=22050, n_fft=1024, n_mels=80, hop_length=256, win_length=1024), origin='lower'), plt.show()
        return mel, em

    def __len__(self):
        return len(self.mel_paths)


class MelLoaderCollate:
    """ DataLoader requires all elements of the batch to have the same size, so we pad them to 0. """

    def __call__(self, batch):
        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([len(x[0][0]) for x in batch]),
            dim=0, descending=True)
        max_input_len = input_lengths[0]
        mel_padded = torch.FloatTensor(len(batch), len(batch[0][0]), max_input_len)
        mel_padded.zero_()
        emotions = torch.FloatTensor(len(batch), len(batch[0][1]))
        emotions.zero_()

        for i in range(len(ids_sorted_decreasing)):
            mel = batch[ids_sorted_decreasing[i]][0]
            mel_padded[i, :, :mel.size(1)] = mel
            emotions[i] = batch[ids_sorted_decreasing[i]][1]

        return mel_padded.cuda(non_blocking=True), input_lengths[-1], emotions.cuda(non_blocking=True)


class Classifier(torch.nn.Module):
    def __init__(self, n_mel_channels, n_frames, n_emotions, linear_model=True):
        super().__init__()
        self.n_mel_channels = n_mel_channels
        self.n_frames = n_frames
        self.linear_model = linear_model
        if linear_model:
            self.model = torch.nn.Sequential(
                module(n_mel_channels * n_frames, 1024),
                module(1024, 1024),
                module(1024, 256),
                module(256, 64),
                torch.nn.Linear(64, n_emotions),
                torch.nn.Sigmoid(),
                torch.nn.Softmax()
            )
        else:
            self.model = torch.nn.Sequential(
                conv_module(n_mel_channels, 512),
                conv_module(512, 256),
                conv_module(256, 128),
                torch.nn.Conv1d(128, n_emotions, 5, padding=2),
                torch.nn.Flatten(),
                torch.nn.Linear(n_emotions * n_frames, n_emotions),
                torch.nn.Dropout(0.5),
                torch.nn.Sigmoid(),
                torch.nn.Softmax()
            )

    def forward(self, x, smallest_length):
        if smallest_length - self.n_frames - 25 > 0:
            start = np.random.randint(25, smallest_length - self.n_frames)
        else:
            start = 0
        x = x[:, :, start:start + self.n_frames]
        if self.linear_model:
            x = x.reshape(x.size(0), -1)
        return self.model(x)


def load_npy_mels(filepaths_list):
    """
    Save all mel spectrograms as npy files so they can be loaded much faster.
    Load the mel spectrograms as Tacotron 2 does.

    """
    hparams = HParams()
    stft = layers.TacotronSTFT(
        hparams.filter_length, hparams.hop_length, hparams.win_length,
        hparams.n_mel_channels, hparams.sampling_rate, hparams.mel_fmin,
        hparams.mel_fmax)
    min_len = float('inf')
    new_filepaths_lists = []
    for n, filepath in enumerate(filepaths_list):
        progress_bar = tqdm(filepath)
        progress_bar.set_description(f'Loading file {n}/{len(filepaths_list)}')
        new_filepaths_list = []
        for path in tqdm(filepath):
            new_filepaths_list.append(path.split('.')[0] + '.npy')
            if not os.path.exists(path.split('.')[0] + '.npy'):
                audio = load_wav_to_torch(path, stft.sampling_rate)
                audio = audio.unsqueeze(0)
                audio = torch.autograd.Variable(audio, requires_grad=False)
                melspec = stft.mel_spectrogram(audio)
                melspec = torch.squeeze(melspec, 0)
                np.save(path.split('.')[0] + '.npy', melspec.numpy())
                if melspec.shape[1] < min_len:
                    min_len = melspec.shape[1]
        new_filepaths_lists.append(new_filepaths_list)

    return new_filepaths_lists


def prepare_data(vesus_path, use_intended_labels, batch_size):
    train_filepaths, train_speakers, train_emotions = load_vesus('filelists/vesus_train.txt', vesus_path,
                                                                 use_intended_labels=use_intended_labels,
                                                                 use_text=False)
    val_filepaths, val_speakers, val_emotions = load_vesus('filelists/vesus_val.txt', vesus_path,
                                                           use_intended_labels=use_intended_labels, use_text=False)
    test_filepaths, test_speakers, test_emotions = load_vesus('filelists/vesus_test.txt', vesus_path,
                                                              use_intended_labels=use_intended_labels, use_text=False)
    train_filepaths, val_filepaths, test_filepaths = load_npy_mels([train_filepaths, val_filepaths, test_filepaths])

    train_loader = DataLoader(MelLoader(train_filepaths, train_emotions), num_workers=0, shuffle=True,
                              batch_size=batch_size, pin_memory=False, drop_last=True, collate_fn=MelLoaderCollate())

    val_loader = DataLoader(MelLoader(val_filepaths, train_emotions), num_workers=0,
                            shuffle=False, batch_size=batch_size, pin_memory=False, collate_fn=MelLoaderCollate())
    test_loader = MelLoader(test_filepaths, train_emotions)

    return train_loader, val_loader, test_loader


def train(vesus_path, use_intended_labels, epochs, learning_rate, batch_size, n_frames):
    train_loader, val_loader, test_loader = prepare_data(vesus_path, use_intended_labels, batch_size)
    hparams = HParams()
    model = Classifier(hparams.n_mel_channels, n_frames, 5).cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=hparams.weight_decay)
    best_val_loss = float('inf')
    last_path = None
    for epoch in range(epochs):
        train_epoch(epoch, epochs, model, optimizer, train_loader, use_intended_labels)
        val_loss = val_epoch(model, val_loader, use_intended_labels)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            path = f"{wandb.run.dir}/epoch={epoch}-val_loss={val_loss}.ckpt"
            torch.save({'epoch': epoch,
                        'state_dict': model.state_dict()}, path)
            if last_path is not None:
                os.remove(last_path)
            last_path = path


def train_epoch(epoch, epochs, model, optimizer, train_loader, use_intended_labels):
    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader))
    model.train()
    for i, batch in progress_bar:
        model.zero_grad()
        x, smallest_length, y = batch
        y_pred = model(x, smallest_length).squeeze(-1)
        if use_intended_labels:
            loss = torch.nn.BCELoss()(y_pred, y)
        else:
            loss = torch.nn.MSELoss()(y_pred, y)
        progress_bar.set_description(f'Epoch {epoch + 1}/{epochs}. Iter {i}/{len(train_loader)}. Loss = {loss}')
        wandb.log({'Train Loss': loss, 'Epoch/Iter': epoch})
        loss.backward()
        optimizer.step()


def val_epoch(model, val_loader, use_intended_labels):
    model.eval()
    with torch.no_grad():
        val_loss = 0
        dominant_emotion_pred = 0
        for batch in val_loader:
            x, smallest_length, y = batch
            y_pred = model(x, smallest_length).squeeze(-1)
            dominant_emotions = 0
            for i in range(len(y)):
                dominant_emotions += int(torch.argmax(y_pred[i]) == torch.argmax(y[i]))
            dominant_emotion_pred += dominant_emotions / len(y)
            if use_intended_labels:
                loss = torch.nn.BCELoss()(y_pred, y)
            else:
                loss = torch.nn.MSELoss()(y_pred, y)
            val_loss += loss
        val_loss /= len(val_loader)
        wandb.log({'Validation Loss': val_loss, 'Dominant emotion prediction': dominant_emotion_pred / len(val_loader)})
        print(f"{100 * dominant_emotion_pred / len(val_loader):.2f} % of dominant emotions where predicted.")
    return val_loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--vesus_path', type=str, required=True, help='Path to audio files')
    parser.add_argument('--use_intended_labels', action='store_true', help='Use intended emotions instead of voted')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size, recommended to use a small one even if it is smaller.')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--n_frames', type=int, default=30, help='Number of frames to use for classification')
    # dryrun
    args = parser.parse_args()
    name = f'{args.batch_size}bs-{args.n_frames}nFrames-{args.lr}LR' \
           f'{"-intendedLabels" if args.use_intended_labels else ""}'
    wandb.init(project="Classifier", config=args, name=name)

    train(args.vesus_path, args.use_intended_labels, args.epochs, args.lr, args.batch_size, args.n_frames)
