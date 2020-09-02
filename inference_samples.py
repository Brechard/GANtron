import argparse
import os
import sys

# matplotlib.use('GTK3')
import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm

from hparams import HParams
from text import text_to_sequence
from train import load_model

sys.path.append('WaveGlow/')


def load_GANtron(path):
    hparams = HParams(args.hparams)
    hparams.add_params(args)

    model, _ = load_model(hparams)
    model.load_state_dict(torch.load(path)['state_dict'])
    model.cuda().eval()
    return model, hparams


def load_waveglow(path):
    waveglow = torch.load(path)['model']
    waveglow.cuda().eval().half()
    for k in waveglow.convinv:
        k.float()
    return waveglow


def generate_audio(waveglow, mel_spectrogram):
    with torch.no_grad():
        audio = waveglow.infer(mel_spectrogram.half(), sigma=0.666)
    return audio


def force_style_emotions(gantron, input_sequence, output_path, speaker, force_emotions, force_style, style_shape=None,
                         n_styles=6, n_samples_styles=20):
    """
    Inference a given number of samples where the style or the emotion is forced.

    Args:
        gantron: GANtron model to use for inference.
        input_sequence: Input sequence to inference.
        output_path: Shape of the style that will be forced.
        speaker: Speaker to use.
        force_emotions: Flag to force the emotions.
        force_style: Flag to force the style.
        style_shape: Folder path to save the inferred samples.
        n_styles: Number of styles/emotions to force.
        n_samples_styles: Number of samples to inference per style.

    Returns:
        None
    """
    emotions, styles = None, None
    if force_emotions:
        emotions = [
                       # [Neutral, Angry, Happy, Sad, Fearful]
                       torch.FloatTensor([[0.6, 0, 0, 0, 0]]).cuda(),
                       torch.FloatTensor([[0, 0.7, 0, 0, 0]]).cuda(),
                       torch.FloatTensor([[0, 0, 0.5, 0, 0]]).cuda(),
                       torch.FloatTensor([[0, 0, 0, 0.8, 0]]).cuda(),
                       torch.FloatTensor([[0, 0, 0, 0, 0.75]]).cuda()
                   ] + [torch.rand(1, 5).cuda() for i in range(n_styles - 3)]
    if force_style:
        styles = [
                     torch.zeros(1, style_shape[0], style_shape[1]).cuda(),
                     torch.ones(1, style_shape[0], style_shape[1]).cuda() * 0.5,
                     torch.ones(1, style_shape[0], style_shape[1]).cuda(),
                 ] + [torch.rand(1, 1, style_shape[1]).repeat_interleave(style_shape[0], dim=1).cuda() for i in
                      range(n_styles - 3)]
    for st in tqdm(range(n_styles)):
        for i in range(n_samples_styles):
            style, emotion = None, None
            if styles is not None:
                style = styles[st]
            if emotions is not None:
                emotion = emotions[st]
            mel_outputs, mel_outputs_postnet, _, alignments = gantron.inference(input_sequence, style,
                                                                                emotions=emotion, speaker=speaker)

            name = ''
            if force_emotions:
                name += f'emotion-{st}-'
            if force_style:
                name += f'style-{st}-'
            np.save(f'{output_path}/{name}{i}.npy', mel_outputs_postnet[0].data.cpu().numpy())


def random_style():
    for i in tqdm(range(args.samples)):
        style = torch.rand(1, 1, hparams.noise_size)
        style = style.repeat_interleave(sequence.size(1), dim=1).cuda()
        emotions = None
        if hparams.use_labels:
            emotions = torch.rand(1, 5).cuda()

        mel_outputs, mel_outputs_postnet, _, alignments = gantron.inference(sequence, style, emotions=emotions,
                                                                            speaker=speaker)
        if waveglow is not None:
            audio = generate_audio(waveglow, mel_outputs_postnet)
            sf.write(f'{args.output_path}/{i}.wav', audio[0].to(torch.float32).data.cpu().numpy(), 22050)
        else:
            np.save(f'{args.output_path}/{i}.npy', mel_outputs_postnet[0].data.cpu().numpy())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--checkpoint_path', type=str, required=True, help='GANtron checkpoint path')
    parser.add_argument('--generate_audio', action='store_true', help='Generate the audio files')
    parser.add_argument('--force', action='store_true', help='Generate with forced styles')
    parser.add_argument('-w', '--waveglow_path', type=str, required=False, help='waveglow checkpoint path')
    parser.add_argument('-o', '--output_path', type=str, required=True, help='Model name to save the ')
    parser.add_argument('--samples', type=int, default=200, help='Number of samples to generate')
    parser.add_argument('--hparams', type=str, required=False, help='comma separated name=value pairs')
    parser.add_argument('--speaker', default=0, type=int, required=False, help='Speaker to use when generating')

    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)

    gantron, hparams = load_GANtron(args.checkpoint_path)
    waveglow = None
    if args.generate_audio:
        waveglow = load_waveglow(args.waveglow_path)

    text = "This voice was generated by a machine"
    sequence = np.array(text_to_sequence(text, ['english_cleaners']))[None, :]
    sequence = torch.autograd.Variable(torch.from_numpy(sequence)).cuda().long()
    speaker = None if args.hparams is None else torch.LongTensor([args.speaker]).cuda()

    if args.force:
        force_style_emotions(gantron, sequence, args.output_path, speaker,
                             force_emotions=hparams.use_labels,
                             force_style=hparams.use_noise,
                             style_shape=[sequence.size(1), hparams.noise_size])
    else:
        random_style()
