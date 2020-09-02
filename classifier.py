import argparse
import os
import time
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
from hparams_classifier import HParams
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
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.criterion = torch.nn.MSELoss()
        if hparams['use_intended_labels']:
            self.criterion = torch.nn.BCEWithLogitsLoss()

        self.val_log = "Validation loss - " + ("Intended" if hparams['use_intended_labels'] else "Calculated")
        if hparams['linear_model']:
            self.model = torch.nn.Sequential(
                module(hparams['n_mel_channels'] * hparams['n_frames'], hparams['model_size']),
                module(hparams['model_size'], hparams['model_size']),
                module(hparams['model_size'], hparams['model_size']),
                torch.nn.Linear(hparams['model_size'], hparams['n_emotions'])
            )
        else:
            flatten_size = int(
                hparams['n_emotions'] * (hparams['n_mel_channels'] / 2 ** 3) * (hparams['n_frames'] / 2 ** 3))
            self.model = torch.nn.Sequential(
                conv_module(1, hparams['model_size']),
                conv_module(hparams['model_size'], hparams['model_size']),
                conv_module(hparams['model_size'], hparams['n_emotions']),
                torch.nn.Flatten(),
                # Divide by 2^3 because of max pool
                torch.nn.Linear(flatten_size, hparams['n_emotions']),
            )

    def forward(self, x, smallest_length):
        if smallest_length - self.hparams['n_frames'] - 25 > 0:
            start = np.random.randint(25, smallest_length.cpu().numpy() - self.hparams['n_frames'])
        elif smallest_length - self.hparams['n_frames'] >= 0:
            start = smallest_length - self.hparams['n_frames']
        else:
            start = 0

        x = x[:, :, start:start + self.hparams['n_frames']]
        if self.hparams['linear_model']:
            x = x.reshape(x.size(0), -1)
        else:
            x = x.unsqueeze(1)

        if isinstance(self.criterion, torch.nn.BCEWithLogitsLoss):
            return self.model(x)

        return torch.nn.Softmax()(self.model(x))

    def inference(self, x):
        """ Shape (Batch, n_mels, n_frames)"""
        if self.hparams['linear_model']:
            x = x.reshape(x.size(0), -1)
        else:
            x = x.unsqueeze(1)
        return torch.nn.Softmax()(self.model(x))

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams['lr'])
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.hparams['epochs'],
                                                                  eta_min=1e-6, last_epoch=-1)
        return [optimizer], [lr_scheduler]

    def training_step(self, batch, batch_idx):
        start = time.perf_counter()
        x, smallest_length, y = batch
        y_hat = self(x, smallest_length).squeeze(-1)
        loss = self.criterion(y_hat, y)
        output = {
            'loss': loss,
            'log': {'train_loss': loss, 'duration': time.perf_counter() - start},
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


def load_npy_mels(filepaths_list, hparams):
    """
    Save all mel spectrograms as np files so they can be loaded much faster.
    """
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
                                       sr=hparams.sampling_rate, n_fft=hparams.n_ftt,
                                       n_mels=hparams.n_mel_channels, hop_length=hparams.hop_length),
        ref=np.max)
    np.save(path.split('.')[0] + '.npy', melspec)


def prepare_data(vesus_path, hparams):
    use_intended_labels, mel_offset, bs = hparams.use_intended_labels, hparams.mel_offset, hparams.batch_size
    train_filepaths, train_speakers, train_emotions = load_vesus(hparams.training_files, vesus_path,
                                                                 use_intended_labels=use_intended_labels,
                                                                 use_text=False)
    val_filepaths, val_speakers, val_emotions = load_vesus(hparams.validation_files, vesus_path,
                                                           use_intended_labels=use_intended_labels, use_text=False)
    test_filepaths, test_speakers, test_emotions = load_vesus(hparams.test_files, vesus_path,
                                                              use_intended_labels=use_intended_labels, use_text=False)
    train_filepaths, val_filepaths, test_filepaths = load_npy_mels([train_filepaths, val_filepaths, test_filepaths],
                                                                   hparams)

    train_loader = DataLoader(MelLoader(train_filepaths, train_emotions, mel_offset), num_workers=0, shuffle=True,
                              batch_size=bs, pin_memory=False, drop_last=True, collate_fn=MelLoaderCollate())

    val_loader = DataLoader(MelLoader(val_filepaths, val_emotions, mel_offset), num_workers=0,
                            shuffle=False, batch_size=bs, pin_memory=False, collate_fn=MelLoaderCollate())

    test_loader = MelLoader(test_filepaths, test_emotions, mel_offset)

    return train_loader, val_loader, test_loader


def train(vesus_path, hparams):
    train_loader, val_loader, test_loader = prepare_data(vesus_path, hparams)

    model = Classifier(hparams.__dict__)

    wandb_logger = WandbLogger(project='Classifier', name=name, log_model=True)
    wandb_logger.log_hyperparams(args)
    checkpoint_callback = ModelCheckpoint(filepath=wandb_logger.save_dir + '/{epoch}-{val_loss:.2f}-{acc:.4f}')

    trainer = pl.Trainer(max_epochs=hparams.epochs, gpus=1, logger=wandb_logger, precision=hparams.precision,
                         checkpoint_callback=checkpoint_callback)
    trainer.fit(model, train_loader, val_loader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--vesus_path', type=str, required=False, help='Path to audio files')
    parser.add_argument('--use_intended_labels', type=str2bool, default=True,
                        help='Use intended emotions instead of voted')
    parser.add_argument('--linear_model', type=str2bool, default=False, help='Use linear model or convolutional')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size, recommended to use a small one even if it is smaller.')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--n_frames', type=int, default=40, help='Number of frames to use for classification')
    parser.add_argument('--precision', type=int, default=16, help='Precision 32/16 bits')
    parser.add_argument('--model_size', type=int, default=512, help='Model size')
    parser.add_argument('--mel_offset', type=int, default=20, help='Mel offset when loading the frames')
    parser.add_argument('--hparams', type=str, default='', help='Comma separated name=value pairs')

    args = parser.parse_args()
    name = f'{args.batch_size}bs-{args.n_frames}nFrames-{args.lr}LR' \
           f'-{args.model_size}{"linear" if args.linear_model else "conv"}' \
           f'{"-intendedLabels" if args.use_intended_labels else ""}'
    # wandb.init(project="Classifier", config=args, name=name)
    hp = HParams(args.hparams)
    hp.add_params(args)
    args.vesus_path = 'C:/Users/rodri/Datasets/Vesus/Audio/'
    if not hp.linear_model and hp.n_frames % 8 != 0:
        raise argparse.ArgumentTypeError("Due to the three MaxPool layers, n_frames must be a multiple of 8")

    train(args.vesus_path, hp)
