"""
In order to study a model the steps are:
    1. Inference samples that we force a style or emotion
    2. Compute the wav file with WaveGlow
    3. Extract the mel spectrogram with librosa
    4. Take the values to the range [0, 1] dividing by 80 and summing 1.
    5. Classify the samples with a Classifier trained on VESUS
    6. Compare the classification results with the group they were suppose to belong to.
"""
import argparse
import os

import numpy as np
import pytorch_lightning as pl
import soundfile as sf
import torch
import wandb
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from tqdm import tqdm

from classifier import Classifier, load_npy_mels
from data_utils import MelLoader, MelLoaderCollate
from hparams import HParams
from hparams_classifier import HParams as HPC
from inference_samples import force_style_emotions
from text import text_to_sequence
from train import load_model
from utils import str2bool


def compute_wav(output_path, hparams):
    waveglow = torch.load(hparams.waveglow_path)['model']
    waveglow.cuda().eval().half()
    for k in waveglow.convinv:
        k.float()
    paths = os.listdir(f"{output_path}/GANtronInference/")
    progress_bar = tqdm(paths)
    progress_bar.set_description(f'Genearting wav files')
    mels = []
    new_paths, batch_paths = [], []
    max_len = 0
    sampled = {}

    for p in progress_bar:
        new_path = f"{output_path}/WaveGlowInference/{p.split('.')[0]}.wav"
        if os.path.exists(new_path):
            new_paths.append(new_path)
            continue
        mel_spectrogram = np.load(f"{output_path}/GANtronInference/{p}", allow_pickle=True)
        mels.append(mel_spectrogram)
        batch_paths.append(p)
        if mel_spectrogram.shape[1] > max_len:
            max_len = mel_spectrogram.shape[1]

        if len(mels) == hparams.waveglow_bs or p == paths[-1]:
            batch_paths, mels = generate_audio(batch_paths, hparams, max_len, mels, new_paths, output_path, sampled,
                                               waveglow)
            max_len = 0

    if len(batch_paths) > 0:
        generate_audio(batch_paths, hparams, max_len, mels, new_paths, output_path, sampled, waveglow)

    return new_paths, sampled


def generate_audio(batch_paths, hparams, max_len, mels, new_paths, output_path, sampled, waveglow):
    new_mels = np.zeros((len(mels), hparams.n_mel_channels, max_len))
    for i, mel in enumerate(mels):
        new_mels[i, :, :mel.shape[1]] = mel
    mels = torch.FloatTensor(new_mels).half().cuda()
    with torch.no_grad():
        audios = waveglow.infer(mels, sigma=0.666)
    for i in range(len(audios)):
        new_path = f"{output_path}/WaveGlowInference/{batch_paths[i].split('.')[0]}.wav"
        sf.write(new_path, audios[i].to(torch.float32).data.cpu().numpy(), 22050)
        group = batch_paths[i].split('.')[0].split('-')[0]
        if group not in sampled:
            sampled[group] = [
                wandb.Audio(audios[i].to(torch.float32).data.cpu().numpy(), caption=f'Group = {group} - 0',
                            sample_rate=22050)]
        elif len(sampled[group]) == 1:
            sampled[group].append(
                wandb.Audio(audios[i].to(torch.float32).data.cpu().numpy(), caption=f'Group = {group} - 1',
                            sample_rate=22050))

        new_paths.append(new_path)
    mels = []
    batch_paths = []
    return batch_paths, mels


def get_filepath_label_by_index_list(filepaths, labels, idx_list):
    return list(map(filepaths.__getitem__, idx_list)), list(map(labels.__getitem__, idx_list))


def inference_samples(output_path, hparams, text):
    sequence = np.array(text_to_sequence(text, ['english_cleaners']))[None, :]
    sequence = torch.autograd.Variable(torch.from_numpy(sequence)).cuda().long()
    speaker = None if args.hparams is None else torch.LongTensor([hparams.speaker]).cuda()

    gantron, _ = load_model(hparams)
    gantron.load_state_dict(torch.load(hparams.gantron_path)['state_dict'])
    gantron.cuda().eval()
    force_emotions = hparams.force_emotions if hasattr(hparams, 'force_emotions') else hparams.use_labels
    force_noise = hparams.force_noise if hasattr(hparams, 'force_noise') else hparams.use_noise

    force_style_emotions(gantron, input_sequence=sequence, output_path=f"{output_path}/GANtronInference/",
                         n_groups=hparams.n_groups, speaker=speaker, force_emotions=force_emotions,
                         force_style=force_noise, simple_name=True, n_samples_styles=hparams.samples,
                         style_shape=[sequence.size(1), hparams.noise_size])


def prepare_data(file_paths, n_groups):
    labels = np.zeros((len(file_paths), n_groups))
    for i, filepath in enumerate(file_paths):
        filename = filepath.split('/')[-1].split('.')[0]
        group, id = filename.split('-')
        label = np.zeros(n_groups)
        label[int(group)] = 1
        labels[i] = label

    idxs = list(range(len(file_paths)))
    np.random.shuffle(idxs)

    val_lim = int(0.85 * len(file_paths))
    test_lim = val_lim + int(0.05 * len(file_paths))
    train_paths, train_groups = list(map(file_paths.__getitem__, idxs[:val_lim])), labels[idxs[:val_lim]]
    val_paths, val_groups = list(map(file_paths.__getitem__, idxs[val_lim:test_lim])), labels[idxs[val_lim:test_lim]]
    test_paths, test_groups = list(map(file_paths.__getitem__, idxs[test_lim:])), labels[idxs[test_lim:]]

    return train_paths, train_groups, val_paths, val_groups, test_paths, test_groups


def study_model(output_path, hparams, text):
    inference_samples(output_path, hparams, text)
    files_paths, sampled = compute_wav(output_path, hparams)
    files_paths = load_npy_mels([files_paths], hparams)
    train_classifier(output_path, files_paths[0], hparams.n_groups, hparams.notes, sampled)


def train_classifier(output_path, files_paths, n_groups, notes, sampled=None):
    hparams_classifier = HPC()
    hparams_classifier.n_emotions = n_groups
    classifier = Classifier(hparams_classifier)
    train_filepaths, train_groups, val_filepaths, val_groups, test_filepaths, test_groups = prepare_data(files_paths,
                                                                                                         n_groups)
    train_loader = DataLoader(
        MelLoader(train_filepaths, train_groups, hparams_classifier.mel_offset, hparams_classifier.max_noise),
        num_workers=0, shuffle=True, batch_size=hparams_classifier.batch_size, pin_memory=False, drop_last=True,
        collate_fn=MelLoaderCollate())

    val_loader = DataLoader(
        MelLoader(val_filepaths, val_groups, hparams_classifier.mel_offset, hparams_classifier.max_noise),
        num_workers=0, shuffle=False, batch_size=hparams_classifier.batch_size, pin_memory=False,
        collate_fn=MelLoaderCollate())

    test_loader = DataLoader(
        MelLoader(test_filepaths, test_groups, hparams_classifier.mel_offset, hparams_classifier.max_noise),
        num_workers=0, shuffle=False, batch_size=hparams_classifier.batch_size, pin_memory=False,
        collate_fn=MelLoaderCollate())

    name = output_path.split('/')[-1]
    if name is None or name == '':
        name = output_path.split('/')[-2]

    wandb_logger = WandbLogger(project='Study models', name=name, log_model=True)
    wandb_logger.log_hyperparams(args)
    wandb_logger.experiment.notes = notes
    if sampled is not None:
        audios = []
        for i in list(sampled.values()):
            audios.extend(i)
        wandb_logger.experiment.log({'Audios': audios})
    checkpoint_callback = ModelCheckpoint(filepath=wandb_logger.save_dir + '/{epoch}-{val_loss:.2f}-{acc:.4f}')

    trainer = pl.Trainer(max_epochs=hparams_classifier.epochs, gpus=1, logger=wandb_logger,
                         precision=hparams_classifier.precision, checkpoint_callback=checkpoint_callback)
    trainer.fit(classifier, train_loader, val_loader)
    result = trainer.test(test_dataloaders=test_loader)
    tot_loss = 0
    for res in result:
        tot_loss += res['test_loss']
    print(f'Test results: {tot_loss / len(result)}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gantron_path', type=str, required=True, help='GANtron checkpoint path')
    parser.add_argument('-w', '--waveglow_path', type=str, required=True, help='WaveGlow checkpoint path')
    parser.add_argument('-o', '--output_path', type=str, required=True, help='Folder to save the comparison')
    parser.add_argument('--samples', type=int, default=10, help='Number of samples to generate')
    parser.add_argument('--waveglow_bs', type=int, default=1,
                        help='Batch size to use waveglow faster. Be careful with it since if audios are not of the '
                             'same size it will generate noise at the end of the file')
    parser.add_argument('--hparams', type=str, required=False, help='comma separated name=value pairs')
    parser.add_argument('--notes', type=str, default='', help='Notes to include in the W&B run.')
    parser.add_argument('--speaker', default=0, type=int, required=False, help='Speaker to use when generating')
    parser.add_argument('--n_groups', default=6, type=int, required=False,
                        help='Number of different groups to generate and classify.')
    parser.add_argument('--force_emotions', default=None, type=str2bool, help='Force using/not labels when generating')
    parser.add_argument('--force_noise', default=None, type=str2bool, help='Force using/not noise when generating')

    args = parser.parse_args()
    os.makedirs(args.output_path, exist_ok=True)
    for folder in ['GANtronInference', 'WaveGlowInference', 'ClassifierOutput']:
        os.makedirs(f'{args.output_path}/{folder}', exist_ok=True)

    hp = HParams()
    hp.add_params(args)

    study_model(args.output_path, hp, text="Emotional speech synthesis")
