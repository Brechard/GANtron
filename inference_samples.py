import argparse
import sys

from tqdm import tqdm

# matplotlib.use('GTK3')
sys.path.append('waveglow/')
import numpy as np
import torch

from hparams import HParams
from train import load_model
from text import text_to_sequence
import soundfile as sf
import os

def load_GANtron(path):
    hparams = HParams()
    hparams.sampling_rate = 22050

    model, _ = load_model(hparams)
    model.load_state_dict(torch.load(path)['state_dict'])
    model.cuda().eval()
    return model


def load_waveglow(path):
    waveglow = torch.load(path)['model']
    waveglow.cuda().eval().half()
    for k in waveglow.convinv:
        k.float()
    return waveglow


def generate(model, waveglow, style):
    mel_outputs, mel_outputs_postnet, _, alignments = model.inference(sequence, style)
    with torch.no_grad():
        audio = waveglow.infer(mel_outputs_postnet.half(), sigma=0.666)
    return audio


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--checkpoint_path', type=str, required=True, help='GANtron checkpoint path')
    parser.add_argument('-w', '--waveglow_path', type=str, required=True, help='waveglow checkpoint path')
    parser.add_argument('-o', '--output_path', type=str, required=True, help='Model name to save the ')
    parser.add_argument('--noise_size', type=int, required=True, help='Number of noise inputs')
    parser.add_argument('--samples', type=int, default=1000, help='Number of samples to generate')

    args = parser.parse_args()
    os.makedirs(args.output_path, exist_ok=True)

    gantron = load_GANtron(args.checkpoint_path)
    waveglow = load_waveglow(args.waveglow_path)

    text = "This voice was generated by a machine"
    sequence = np.array(text_to_sequence(text, ['english_cleaners']))[None, :]
    sequence = torch.autograd.Variable(torch.from_numpy(sequence)).cuda().long()

    for i in tqdm(range(args.samples)):
        style = torch.rand(1, 1, args.noise_size)
        style = style.repeat_interleave(sequence.size(1), dim=1)
        audio = generate(gantron, waveglow, style.cuda())
        sf.write(f'{args.output_path}/{i}.wav', audio[0].to(torch.float32).data.cpu().numpy(), 22050)