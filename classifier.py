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
        torch.nn.LayerNorm(size_out),
        torch.nn.Dropout(0.5),
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
        return self.get_mel(self.mel_paths[self.indexes[index]]), torch.FloatTensor(self.emotions[self.indexes[index]])

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
    def __init__(self, n_mel_channels, n_frames, n_emotions):
        super().__init__()
        self.n_mel_channels = n_mel_channels
        self.n_frames = n_frames
        self.model = torch.nn.Sequential(
            module(n_mel_channels * n_frames, 256),
            module(256, 128),
            module(128, 64),
            module(64, n_emotions)
        )

    def forward(self, x, smallest_length):
        start = np.random.randint(0, smallest_length - self.n_frames)
        x = x[:, :, start:start + self.n_frames].reshape(x.size(0), -1)
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
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=hparams.weight_decay)
    for epoch in range(epochs):
        train_epoch(epoch, model, optimizer, train_loader)
        val_epoch(model, val_loader)


def train_epoch(epoch, model, optimizer, train_loader):
    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader))
    model.train()
    for i, batch in progress_bar:
        model.zero_grad()
        x, smallest_length, y = batch
        y_pred = model(x, smallest_length).squeeze(-1)
        loss = torch.nn.MSELoss()(y, y_pred)
        progress_bar.set_description(f'Epoch {epoch + 1}/epochs. Iter {i}/{len(train_loader)}. Loss = {loss}')
        wandb.log({'Train Loss': loss})
        loss.backward()
        optimizer.step()


def val_epoch(model, val_loader):
    model.eval()
    with torch.no_grad():
        val_loss = 0
        for batch in val_loader:
            x, smallest_length, y = batch
            y_pred = model(x, smallest_length).squeeze(-1)
            loss = torch.nn.MSELoss()(y, y_pred)
            val_loss += loss
        wandb.log({'Validation Loss': val_loss / len(val_loader)})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--vesus_path', type=str, required=True, help='Path to audio files')
    parser.add_argument('--use_intended_labels', action='store_true', help='Use intended emotions instead of voted')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size, recommended to use a small one even if it is smaller.')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--n_frames', type=int, default=40, help='Number of frames to use for classification')
    # dryrun
    args = parser.parse_args()
    wandb.init(project="Classifier", config=args)

    train(args.vesus_path, args.use_intended_labels, args.epochs, args.lr, args.batch_size, args.n_frames)
