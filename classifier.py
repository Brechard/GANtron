import argparse
import os
import time
import warnings
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
from utils import load_vesus, load_cremad_ravdess, str2bool


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

    module = torch.nn.Sequential(
        torch.nn.Conv2d(size_in, size_out, kernel_size=kernel_size, padding=padding, dilation=dilation),
        torch.nn.BatchNorm2d(size_out),
        torch.nn.Dropout(0.5),
        torch.nn.LeakyReLU(0.1)
    )
    if avg_pool > 0:
        module.add_module('MaxPool', torch.nn.AvgPool2d((avg_pool, avg_pool)))
    return module


def get_random_start(offset, length):
    if length - offset > 0:
        start = torch.randint(offset, length, (1,))
    elif length > 0:
        start = torch.randint(0, length, (1,))
    else:
        start = 0
    return start


class Classifier(pl.LightningModule, ABC):
    def __init__(self, hparams):
        super().__init__()
        if type(hparams) == HParams:
            hparams = hparams.__dict__
        self.hparams = hparams
        self.criterion = torch.nn.MSELoss()
        if hparams['use_labels'] == 'one' or hparams['use_labels'] == 'intended':
            self.criterion = torch.nn.BCEWithLogitsLoss()

        self.val_log = "Validation loss - "
        if hparams['use_labels'] == 'intended':
            self.val_log += "Intended"
        elif hparams['use_labels'] == 'multi':
            self.val_log += "Multi"
        else:
            self.val_log += "One"

        if hparams['linear_model']:
            self.flatten_size = hparams['n_mel_channels'] * hparams['n_frames']
            self.model = torch.nn.Sequential(
                module(self.flatten_size, hparams['model_size']),
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
                conv_module(hparams['model_size'], hparams['model_size']),
                conv_module(hparams['model_size'], hparams['n_emotions'], avg_pool=0),
                torch.nn.Flatten(),
                # Divide by 2^3 because of max pool
                torch.nn.Linear(flatten_size, hparams['n_emotions']),
            )

    def forward(self, x, lengths):
        new_x = torch.zeros_like(x[:, :, :self.hparams['n_frames']])
        for pos, lenght in enumerate(lengths):
            start = get_random_start(self.hparams['mel_offset'], lenght - self.hparams['n_frames'])
            new_x[pos] = x[pos, :, start:start + self.hparams['n_frames']]

        x = new_x
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
            if x.size(1) != self.flatten_size:
                warnings.warn('Input size does not fit the model, we will return the average of a sliding window')
                n_frames_exact = int(x.size(1) / self.flatten_size) * self.flatten_size
                new_x = x[:, :n_frames_exact].reshape(-1, self.flatten_size)
                if n_frames_exact != x.size(1):
                    new_x = torch.cat([new_x, x[:, -self.flatten_size:]])
                x = new_x
        else:
            if x.size(-1) % self.hparams['n_frames'] != 0:
                # The input cannot be divided exactly so there will be some overlapping between the last two windows.
                warnings.warn(
                    'Input size does not fit the model, we will return the result of a sliding window as a list')
                n_frames_exact = int(x.size(-1) / self.hparams['n_frames']) * self.hparams['n_frames']
                new_x = x[:, :, :n_frames_exact].reshape(-1, self.hparams['n_mel_channels'],
                                                         self.hparams['n_frames'])
                new_x = torch.cat([new_x, x[:, :, -self.hparams['n_frames']:]])
                return torch.nn.Softmax()(self.model(new_x.unsqueeze(1)))

            x = x.reshape(-1, 1, self.hparams['n_mel_channels'], self.hparams['n_frames'])
        return torch.nn.Softmax()(self.model(x))

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams['lr'])
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.hparams['epochs'],
                                                                  eta_min=1e-6, last_epoch=-1)
        return [optimizer], [lr_scheduler]

    def training_step(self, batch, batch_idx):
        start = time.perf_counter()
        x, lenghts, y, paths = batch
        y_hat = self(x, lenghts).squeeze(-1)
        loss = self.criterion(y_hat, y)
        output = {
            'loss': loss,
            'log': {'train_loss': loss, 'duration': time.perf_counter() - start},
        }
        return output

    def validation_step(self, batch, batch_idx):
        x, lenghts, y, paths = batch
        y_hat = self(x, lenghts).squeeze(-1)
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
        x, lenghts, y, paths = batch
        y_hat = self(x, lenghts).squeeze(-1)
        loss = self.criterion(y_hat, y)
        output = {
            'test_loss': loss,
            'progress_bar': {'test_loss': loss},
        }
        return output


def load_npy_mels(filepaths_lists, hparams, file_format='.wav'):
    """
    Save all mel spectrograms as np files so they can be loaded much faster.

    Args:
        filepaths_lists: List with the filepaths to the files containing the filepaths for train, val and test
        hparams: Hyperparameters for loading the mel spectrograms
        file_format: Format of the file, defaults to .wav

    Returns:
        List with the new filepaths
    """

    new_filepaths_lists = []
    for n, filepath in enumerate(filepaths_lists):
        progress_bar = tqdm(filepath)
        progress_bar.set_description(f'Loading file {n}/{len(filepaths_lists)}')
        new_filepaths_list = []
        a = 0
        for path in progress_bar:
            new_path = path.split(file_format)[0] + '.npy'
            if not os.path.exists(new_path):
                load_mel(path, hparams, new_path)
            new_filepaths_list.append(new_path)
            a += 1
        new_filepaths_lists.append(new_filepaths_list)

    return new_filepaths_lists


def load_mel(path, hparams, new_path):
    melspec = librosa.power_to_db(
        librosa.feature.melspectrogram(librosa.load(path)[0],
                                       sr=hparams.sampling_rate, n_fft=hparams.n_ftt,
                                       n_mels=hparams.n_mel_channels, hop_length=hparams.hop_length),
        ref=np.max)
    np.save(new_path, melspec)


def load_files(files, audio_path, use_labels):
    filepaths, _, emotions = load_vesus(files[0], audio_path + '/VESUS/Audio/',
                                        use_labels=use_labels, use_text=False)
    cremad_file, cremad_em = load_cremad_ravdess(files[1], audio_path + '/Crema-D/AudioWAV/',
                                                 use_labels=use_labels, crema=True)
    filepaths.extend(cremad_file)
    emotions.extend(cremad_em)
    ravdess_file, ravdess_em = load_cremad_ravdess(files[2], audio_path + '/RAVDESS/Speech/',
                                                   use_labels=use_labels, crema=False)
    filepaths.extend(ravdess_file)
    emotions.extend(ravdess_em)
    return filepaths, emotions


def load_extension(extend_path, use_labels, train_filepaths, train_emotions, val_filepaths, val_emotions,
                   test_filepaths, test_emotions):
    function = (lambda x: 1 if float(x) > 0 else 0) if use_labels in ['one', 'intended'] else float
    for file in os.listdir(extend_path):
        if '.wav' not in file or file[0] == '5':
            continue
        label = np.array([function(i) for i in file.split('.wav')[0].split('-')[-1].split(',')])
        if np.random.random() < 0.8:
            train_filepaths.append(extend_path + file)
            train_emotions.append(label)
        elif np.random.random() < 0.25:
            val_filepaths.append(extend_path + file)
            val_emotions.append(label)
        else:
            test_filepaths.append(extend_path + file)
            test_emotions.append(label)


def prepare_data(audio_path, hparams, extend_path):
    max_noise, mel_offset, bs = hparams.max_noise, hparams.mel_offset, hparams.batch_size
    train_filepaths, train_emotions = load_files(hparams.training_files, audio_path, hparams.use_labels)
    val_filepaths, val_emotions = load_files(hparams.validation_files, audio_path, hparams.use_labels)
    test_filepaths, test_emotions = load_files(hparams.test_files, audio_path, hparams.use_labels)
    if extend_path is not None:
        load_extension(extend_path, hparams.use_labels, train_filepaths, train_emotions, val_filepaths, val_emotions,
                       test_filepaths, test_emotions)

    train_filepaths, val_filepaths, test_filepaths = load_npy_mels([train_filepaths, val_filepaths, test_filepaths],
                                                                   hparams)

    train_loader = DataLoader(MelLoader(train_filepaths, train_emotions, mel_offset, max_noise), num_workers=0,
                              shuffle=True, batch_size=bs, pin_memory=False, drop_last=True,
                              collate_fn=MelLoaderCollate())

    val_loader = DataLoader(MelLoader(val_filepaths, val_emotions, mel_offset, max_noise), num_workers=0,
                            shuffle=False, batch_size=bs, pin_memory=False, collate_fn=MelLoaderCollate())

    test_loader = MelLoader(test_filepaths, test_emotions, mel_offset, max_noise)

    return train_loader, val_loader, test_loader


def train(audio_path, hparams, extend_path):
    train_loader, val_loader, test_loader = prepare_data(audio_path, hparams, extend_path)

    model = Classifier(hparams.__dict__)

    wandb_logger = WandbLogger(project='Classifier', name=name, log_model=True, tags=hparams.model_version)
    wandb_logger.log_hyperparams(args)
    checkpoint_callback = ModelCheckpoint(filepath=wandb_logger.save_dir + '/{epoch}-{val_loss:.2f}-{acc:.4f}')

    trainer = pl.Trainer(max_epochs=hparams.epochs, gpus=1, logger=wandb_logger, precision=hparams.precision,
                         checkpoint_callback=checkpoint_callback)
    trainer.fit(model, train_loader, val_loader)
    result = trainer.test(test_dataloaders=test_loader)
    tot_loss = 0
    for res in result:
        tot_loss += res['test_loss']
    print(f'Test results: {tot_loss / len(result)}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio_path', type=str, required=False, help='Path to audio files')
    parser.add_argument('--use_labels', type=str, default='one', help="can be either \'one\' (maximum of the voted), "
                                                                      "\'intended\' (what actor was supposed to do) or"
                                                                      "\'multi\' (result of calculated emotions)")
    parser.add_argument('--linear_model', type=str2bool, default=True, help='Use linear model or convolutional')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size, recommended to use a small one even if it is smaller.')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--n_frames', type=int, default=80, help='Number of frames to use for classification')
    parser.add_argument('--precision', type=int, default=32, help='Precision 32/16 bits')
    parser.add_argument('--model_size', type=int, default=512, help='Model size')
    parser.add_argument('--mel_offset', type=int, default=20, help='Mel offset when loading the frames')
    parser.add_argument('--max_noise', type=int, default=3, help='Maximum noise to add to the dataset')
    parser.add_argument('--hparams', type=str, default=None, help='Comma separated name=value pairs')
    parser.add_argument('--extend_path', type=str, default=None,
                        help='Path to the dataset that will be used to extend the current ones')

    args = parser.parse_args()
    hp = HParams()
    hp.add_params(args)
    if args.hparams is not None:
        hp.add_params(args.hparams)

    name = f'v{hp.model_version}-{hp.batch_size}bs-{hp.n_frames}nFrames-{hp.lr}LR' \
           f'-{hp.model_size}{"linear" if hp.linear_model else "conv"}' \
           f'-{hp.use_labels}{("-ext_" + args.extend_path.split("/")[-2]) if args.extend_path is not None else ""}'

    print('\033[94m', f'Run {name} started', '\033[0m')
    args.audio_path = 'C:/Users/rodri/Datasets/'
    if not hp.linear_model and hp.n_frames % 8 != 0:
        raise argparse.ArgumentTypeError("Due to the three MaxPool layers, n_frames must be a multiple of 8")

    train(args.audio_path, hp, args.extend_path)
