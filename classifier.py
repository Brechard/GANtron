import argparse
import os
from abc import ABC

import librosa
import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_utils import MelLoader, MelLoaderCollate
from hparams import HParams
from utils import load_vesus, str2bool


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
        torch.nn.Dropout(0.5),
        torch.nn.LeakyReLU(0.1),
        torch.nn.AvgPool2d((avg_pool, avg_pool))
    )


class Classifier(pl.LightningModule, ABC):
    def __init__(self, n_mel_channels, n_frames, n_emotions, criterion, lr, linear_model, model_size, intended_labels,
                 epochs):
        super().__init__()
        self.n_mel_channels = n_mel_channels
        self.n_frames = n_frames
        self.criterion = criterion
        self.lr = lr
        self.linear_model = linear_model
        self.epochs = epochs
        self.val_log = "Validation loss - " + ("Intended" if intended_labels else "Calculated")
        if linear_model:
            self.model = torch.nn.Sequential(
                module(n_mel_channels * n_frames, model_size),
                module(model_size, model_size),
                module(model_size, model_size),
                torch.nn.Linear(model_size, n_emotions)
            )
        else:
            self.model = torch.nn.Sequential(
                conv_module(1, model_size),
                conv_module(model_size, model_size),
                conv_module(model_size, n_emotions),
                torch.nn.Flatten(),
                # Divide by 2^3 because of max pool
                torch.nn.Linear(int(n_emotions * (n_mel_channels / 2 ** 3) * (n_frames / 2 ** 3)), n_emotions),
            )

    def forward(self, x, smallest_length):
        if smallest_length - self.n_frames - 25 > 0:
            start = np.random.randint(25, smallest_length.cpu().numpy() - self.n_frames)
        elif smallest_length - self.n_frames >= 0:
            start = smallest_length - self.n_frames
        else:
            start = 0

        x = x[:, :, start:start + self.n_frames]
        if self.linear_model:
            x = x.reshape(x.size(0), -1)
        else:
            x = x.unsqueeze(1)

        if isinstance(self.criterion, torch.nn.BCEWithLogitsLoss):
            return self.model(x)

        return torch.nn.Softmax()(self.model(x))

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.epochs,
                                                                  eta_min=1e-8, last_epoch=-1)
        return [optimizer], [lr_scheduler]

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
            self.val_log: avg_loss,
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
    Save all mel spectrograms as np files so they can be loaded much faster.
    """
    hparams = HParams()
    new_filepaths_lists = []
    for n, filepath in enumerate(filepaths_list):
        progress_bar = tqdm(filepath)
        progress_bar.set_description(f'Loading file {n}/{len(filepaths_list)}')
        new_filepaths_list = []
        for path in tqdm(filepath):
            new_filepaths_list.append(path.split('.')[0] + '.npy')
            if not os.path.exists(path.split('.')[0] + '.npy'):
                load_mel(path.split('.')[0] + '.npy', hparams)
        new_filepaths_lists.append(new_filepaths_list)

    return new_filepaths_lists


def load_mel(path, hparams):
    melspec = librosa.power_to_db(
        librosa.feature.melspectrogram(librosa.load(path)[0],
                                       sr=hparams.sampling_rate, n_fft=hparams.filter_length,
                                       n_mels=hparams.n_mel_channels, hop_length=hparams.hop_length),
        ref=np.max)
    np.save(path.split('.')[0] + '.npy', melspec)


def prepare_data(vesus_path, use_intended_labels, batch_size, mel_offset):
    train_filepaths, train_speakers, train_emotions = load_vesus('filelists/vesus_train.txt', vesus_path,
                                                                 use_intended_labels=use_intended_labels,
                                                                 use_text=False)
    val_filepaths, val_speakers, val_emotions = load_vesus('filelists/vesus_val.txt', vesus_path,
                                                           use_intended_labels=use_intended_labels, use_text=False)
    test_filepaths, test_speakers, test_emotions = load_vesus('filelists/vesus_test.txt', vesus_path,
                                                              use_intended_labels=use_intended_labels, use_text=False)
    train_filepaths, val_filepaths, test_filepaths = load_npy_mels([train_filepaths, val_filepaths, test_filepaths])

    train_loader = DataLoader(MelLoader(train_filepaths, train_emotions, mel_offset), num_workers=0, shuffle=True,
                              batch_size=batch_size, pin_memory=False, drop_last=True, collate_fn=MelLoaderCollate())

    val_loader = DataLoader(MelLoader(val_filepaths, val_emotions, mel_offset), num_workers=0,
                            shuffle=False, batch_size=batch_size, pin_memory=False, collate_fn=MelLoaderCollate())

    test_loader = MelLoader(test_filepaths, test_emotions, mel_offset)

    return train_loader, val_loader, test_loader


def train(vesus_path, use_intended_labels, epochs, learning_rate, batch_size, n_frames, name, precision, mel_offset,
          linear_model, model_size):
    train_loader, val_loader, test_loader = prepare_data(vesus_path, use_intended_labels, batch_size, mel_offset)
    criterion = torch.nn.MSELoss()
    if use_intended_labels:
        criterion = torch.nn.BCEWithLogitsLoss()

    hparams = HParams()
    model = Classifier(hparams.n_mel_channels, n_frames, n_emotions=5, criterion=criterion, lr=learning_rate,
                       linear_model=linear_model, model_size=model_size, intended_labels=use_intended_labels,
                       epochs=epochs)

    wandb_logger = WandbLogger(project='Classifier', name=name, log_model=True)
    wandb_logger.log_hyperparams(args)
    checkpoint_callback = ModelCheckpoint(filepath=wandb_logger.save_dir + '/{epoch}-{val_loss:.2f}-{acc:.4f}')

    trainer = pl.Trainer(max_epochs=epochs, gpus=1, logger=wandb_logger, precision=precision,
                         checkpoint_callback=checkpoint_callback)
    trainer.fit(model, train_loader, val_loader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--vesus_path', type=str, required=True, help='Path to audio files')
    parser.add_argument('--use_intended_labels', type=str2bool, default=True,
                        help='Use intended emotions instead of voted')
    parser.add_argument('--linear_model', type=str2bool, default=False, help='Use linear model or convolutional')
    parser.add_argument('--epochs', type=int, default=300, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size, recommended to use a small one even if it is smaller.')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--n_frames', type=int, default=40, help='Number of frames to use for classification')
    parser.add_argument('--precision', type=int, default=16, help='Precision 32/16 bits')
    parser.add_argument('--model_size', type=int, default=512, help='Model size')
    parser.add_argument('--mel_offset', type=int, default=20, help='Mel offset when loading the frames')

    args = parser.parse_args()
    name = f'{args.batch_size}bs-{args.n_frames}nFrames-{args.lr}LR' \
           f'-{args.model_size}{"linear" if args.linear_model else "conv"}' \
           f'{"-intendedLabels" if args.use_intended_labels else ""}'
    # wandb.init(project="Classifier", config=args, name=name)

    train(args.vesus_path, args.use_intended_labels, args.epochs, args.lr, args.batch_size, args.n_frames, name,
          args.precision, args.mel_offset, args.linear_model, args.model_size)
