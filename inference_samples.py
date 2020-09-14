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
                         n_groups=6, n_samples_styles=20, simple_name=False, int_emotions=False, predefined=False,
                         encoder_input=False, max_decoder_steps=500):
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
        n_groups: Number of styles/emotions to force.
        n_samples_styles: Number of samples to inference per style.
        simple_name: If name is simple it will be: groupId-nFile.wav, otherwise it will indicate if it was forced style
        and/or forced emotion.
        int_emotions: Set the emotions as only integers.
        predefined: Flag to use the predefined emotions or to make groups of random values.
        encoder_input: If the input is in the encoder the style is shaped differently.
        max_decoder_steps: Number of maximum steps in GANtron, used to count the number of files that were forced to stop.
    Returns:
        None
    """
    print(f'Saving data in {output_path}')
    emotions, styles = None, None
    max_decoder_steps_reached = 0
    if force_emotions:
        if int_emotions:
            if n_groups > 6:
                raise ValueError('When using emotions as integers there are no more combinations possible than 6.')
            emotions = [
                # [Neutral, Angry, Happy, Sad, Fearful]
                torch.FloatTensor([[1, 0, 0, 0, 0]]).cuda(),
                torch.FloatTensor([[0, 1, 0, 0, 0]]).cuda(),
                torch.FloatTensor([[0, 0, 1, 0, 0]]).cuda(),
                torch.FloatTensor([[0, 0, 0, 1, 0]]).cuda(),
                torch.FloatTensor([[0, 0, 0, 0, 1]]).cuda(),
                torch.FloatTensor([[0, 0, 0, 0, 0]]).cuda()
            ]
        elif predefined:
            emotions = [
                           # [Neutral, Angry, Happy, Sad, Fearful]
                           torch.FloatTensor([[0.6, 0, 0, 0, 0]]).cuda(),
                           torch.FloatTensor([[0, 0.7, 0, 0, 0]]).cuda(),
                           torch.FloatTensor([[0, 0, 0.5, 0, 0]]).cuda(),
                           torch.FloatTensor([[0, 0, 0, 0.8, 0]]).cuda(),
                           torch.FloatTensor([[0, 0, 0, 0, 0.75]]).cuda()
                       ] + [torch.rand(1, 5).cuda() for i in range(n_groups - 5)]
        else:
            emotions = [torch.rand(1, 5).cuda() for i in range(n_groups)]
    if force_style:
        if encoder_input:
            styles = [torch.rand(1, style_shape[1], 1).repeat_interleave(style_shape[0], dim=2).cuda() for i in
                      range(n_groups)]
        else:
            styles = [torch.rand(1, 1, style_shape[1]).repeat_interleave(style_shape[0], dim=1).cuda() for i in
                      range(n_groups)]

    for st in range(n_groups):
        progress_bar = tqdm(range(n_samples_styles))
        progress_bar.set_description(f'Generating group {st + 1} of {n_groups}')
        for i in progress_bar:
            style, emotion = None, None
            if styles is not None:
                style = styles[st]
            if emotions is not None:
                emotion = emotions[st]
            mel_outputs_postnet = gantron.inference(input_sequence, style, emotions=emotion, speaker=speaker)[1]
            if mel_outputs_postnet.shape[-1] == max_decoder_steps:
                max_decoder_steps_reached += 1
            if simple_name:
                name = f'{st}-{i}' + (
                    ('-' + ','.join([str(round(i, 2)) for i in emotion[0].cpu().numpy()])) if force_emotions else '')
            else:
                name = ''
                if force_emotions:
                    name += f'emotion-{st}-'
                if force_style:
                    name += f'style-{st}-'
                name += f'{i}'
            np.save(f'{output_path}/{name}.npy', mel_outputs_postnet[0].data.cpu().numpy())

    return max_decoder_steps_reached


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
