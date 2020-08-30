import argparse
import os
from random import shuffle

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import WandbLogger
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
        torch.nn.Dropout(0.5),
        torch.nn.LeakyReLU(0.1)
    )


def conv_module(size_in, size_out, kernel_size=3, dilation=1, padding=None, avg_pool=2):
    if padding is None:
        assert (kernel_size % 2 == 1)
        padding = int(dilation * (kernel_size - 1) / 2)

    return torch.nn.Sequential(
        torch.nn.Conv2d(size_in, size_out, kernel_size=kernel_size, padding=padding, dilation=dilation),
        torch.nn.BatchNorm2d(size_out),
        torch.nn.Dropout(0.1),
        torch.nn.LeakyReLU(0.1),
        torch.nn.AvgPool2d((avg_pool, avg_pool))
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


class Classifier(pl.LightningModule):
    def __init__(self, n_mel_channels, n_frames, n_emotions, criterion, lr):
        super().__init__()
        self.n_mel_channels = n_mel_channels
        self.n_frames = n_frames
        self.criterion = criterion
        self.lr = lr
        self.model = torch.nn.Sequential(
            module(n_mel_channels * n_frames, 256),
            module(256, 32),
            torch.nn.Linear(32, n_emotions),
            torch.nn.Softmax()
        )

    def forward(self, x, smallest_length):
        if smallest_length - self.n_frames - 25 > 0:
            start = np.random.randint(25, smallest_length.cpu().numpy() - self.n_frames)
        elif smallest_length - self.n_frames >= 0:
            start = smallest_length - self.n_frames
        else:
            start = 0

        x = x[:, :, start:start + self.n_frames].reshape(x.size(0), -1)
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, smallest_length, y = batch
        y_hat = self(x, smallest_length).squeeze(-1)
        loss = self.criterion(y_hat, y)
        output = {
            'loss': loss,
            'progress_bar': {'train_loss': loss},
            'log': {'train_loss': loss},
        }
        return output

    def validation_step(self, batch, batch_idx):
        x, smallest_length, y = batch
        y_hat = self(x, smallest_length).squeeze(-1)
        acc = 0
        for i in range(len(y)):
            acc += int(torch.argmax(y[i]) == torch.argmax(y_hat[i]))

        loss = self.criterion(y_hat, y)
        output = {
            'val_loss': loss,
            'acc': acc / len(y)
        }
        return output

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_acc = np.mean([x['acc'] for x in outputs])
        logs = {
            'val_loss': avg_loss,
            'acc': avg_acc
        }

        return {'avg_val_loss': avg_loss, 'log': logs}

    def test_step(self, batch, batch_idx):
        x, smallest_length, y = batch
        y_hat = self(x, smallest_length).squeeze(-1)
        loss = self.criterion(y_hat, y)
        output = {
            'test_loss': loss,
            'progress_bar': {'test_loss': loss},
        }
        return output


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


def train(vesus_path, use_intended_labels, epochs, learning_rate, batch_size, n_frames, name, precision):
    train_loader, val_loader, test_loader = prepare_data(vesus_path, use_intended_labels, batch_size)
    criterion = torch.nn.MSELoss()
    if use_intended_labels:
        criterion = torch.nn.BCELoss()

    hparams = HParams()
    model = Classifier(hparams.n_mel_channels, n_frames, 5, criterion=criterion, lr=learning_rate)
    wandb_logger = WandbLogger(project='Classifier', name=name, log_model=True)
    wandb_logger.log_hyperparams(args)
    trainer = pl.Trainer(max_epochs=epochs, gpus=1, logger=wandb_logger, precision=precision)
    trainer.fit(model, train_loader, val_loader)


if __name__ == '__main__':
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')


    parser = argparse.ArgumentParser()
    parser.add_argument('--vesus_path', type=str, required=True, help='Path to audio files')
    parser.add_argument('--use_intended_labels', type=str2bool, default=False,
                        help='Use intended emotions instead of voted')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size, recommended to use a small one even if it is smaller.')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--n_frames', type=int, default=80, help='Number of frames to use for classification')
    parser.add_argument('--precision', type=int, default=32, help='Precision 32/16 bits')

    args = parser.parse_args()
    name = f'{args.batch_size}bs-{args.n_frames}nFrames-{args.lr}LR' \
           f'{"-intendedLabels" if args.use_intended_labels else ""}'
    # wandb.init(project="Classifier", config=args, name=name)

    train(args.vesus_path, args.use_intended_labels, args.epochs, args.lr, args.batch_size, args.n_frames, name,
          args.precision)
