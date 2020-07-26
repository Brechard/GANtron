import random

import numpy as np
import torch
import torch.utils.data

import layers
from text import text_to_sequence
from utils import load_wav_to_torch, load_filepaths_and_text, load_vesus


class TextMelLoader(torch.utils.data.Dataset):
    """
        1) loads audio,text pairs
        2) normalizes text and converts them to sequences of one-hot vectors
        3) computes mel-spectrograms from audio files.
    """

    def __init__(self, audiopaths_and_text, hparams, wavs_path):
        self.audiopaths_and_text = load_filepaths_and_text(audiopaths_and_text[0], wavs_path)

        self.vesus = False
        if hparams.vesus_path:
            self.vesus = True
            audiopaths_and_text, speakers, emotions = load_vesus(audiopaths_and_text[1], hparams.vesus_path)
            self.speakers = torch.IntTensor([0] * len(self.audiopaths_and_text) + speakers)
            self.emotions = torch.FloatTensor([[0, 0, 0, 0, 0]] * len(self.audiopaths_and_text) + emotions)
            self.audiopaths_and_text.extend(audiopaths_and_text)

        self.text_cleaners = hparams.text_cleaners
        self.max_wav_value = hparams.max_wav_value
        self.sampling_rate = hparams.sampling_rate
        self.load_mel_from_disk = hparams.load_mel_from_disk
        self.idx = list(range(len(self.audiopaths_and_text)))
        self.stft = layers.TacotronSTFT(
            hparams.filter_length, hparams.hop_length, hparams.win_length,
            hparams.n_mel_channels, hparams.sampling_rate, hparams.mel_fmin,
            hparams.mel_fmax)
        random.seed(hparams.seed)
        random.shuffle(self.idx)

    def get_mel_text_pair(self, audiopath_and_text):
        # separate filename and text
        audiopath, text = audiopath_and_text[0], audiopath_and_text[1]
        text = self.get_text(text)
        mel = self.get_mel(audiopath)
        return text, mel

    def get_mel(self, filename):
        if not self.load_mel_from_disk:
            audio = load_wav_to_torch(filename, self.stft.sampling_rate)
            audio = audio.unsqueeze(0)
            audio = torch.autograd.Variable(audio, requires_grad=False)
            melspec = self.stft.mel_spectrogram(audio)
            melspec = torch.squeeze(melspec, 0)
        else:
            melspec = torch.from_numpy(np.load(filename))
            assert melspec.size(0) == self.stft.n_mel_channels, (
                'Mel dimension mismatch: given {}, expected {}'.format(
                    melspec.size(0), self.stft.n_mel_channels))

        return melspec

    def get_text(self, text):
        text_norm = torch.IntTensor(text_to_sequence(text, self.text_cleaners))
        return text_norm

    def __getitem__(self, index):
        idx = self.idx[index]
        text, mel = self.get_mel_text_pair(self.audiopaths_and_text[idx])
        if self.vesus:
            return text, mel, self.speakers[idx], self.emotions[idx]

        return text, mel

    def __len__(self):
        return len(self.audiopaths_and_text)


class TextMelCollate():
    """ Zero-pads model inputs and targets based on number of frames per step """

    def __init__(self, n_frames_per_step):
        self.n_frames_per_step = n_frames_per_step

    def __call__(self, batch):
        """Collate's training batch from normalized text and mel-spectrogram
        PARAMS
        ------
        batch: [text_normalized, mel_normalized, speaker, emotions]
        """
        # Right zero-pad all one-hot text sequences to max input length
        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([len(x[0]) for x in batch]),
            dim=0, descending=True)
        max_input_len = input_lengths[0]

        text_padded = torch.LongTensor(len(batch), max_input_len)
        text_padded.zero_()
        speaker_ids = torch.FloatTensor(len(batch))
        emotions = torch.FloatTensor(len(batch), 5)
        for i in range(len(ids_sorted_decreasing)):
            text = batch[ids_sorted_decreasing[i]][0]
            text_padded[i, :text.size(0)] = text
            if len(batch[0]) == 4:
                speaker_ids[i] = batch[ids_sorted_decreasing[i]][-2]
                emotions[i] = batch[ids_sorted_decreasing[i]][-1]

        # Right zero-pad mel-spec
        num_mels = batch[0][1].size(0)
        max_target_len = max([x[1].size(1) for x in batch])
        if max_target_len % self.n_frames_per_step != 0:
            max_target_len += self.n_frames_per_step - max_target_len % self.n_frames_per_step
            assert max_target_len % self.n_frames_per_step == 0

        # include mel padded and gate padded
        mel_padded = torch.FloatTensor(len(batch), num_mels, max_target_len)
        mel_padded.zero_()
        gate_padded = torch.FloatTensor(len(batch), max_target_len)
        gate_padded.zero_()
        output_lengths = torch.LongTensor(len(batch))
        for i in range(len(ids_sorted_decreasing)):
            mel = batch[ids_sorted_decreasing[i]][1]
            mel_padded[i, :, :mel.size(1)] = mel
            gate_padded[i, mel.size(1) - 1:] = 1
            output_lengths[i] = mel.size(1)

        return text_padded, input_lengths, mel_padded, gate_padded, speaker_ids, emotions, \
               output_lengths
