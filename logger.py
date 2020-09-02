import random

import torch
import wandb

from plotting_utils import plot_alignment_to_numpy, plot_spectrogram_to_numpy
from plotting_utils import plot_gate_outputs_to_numpy


def log_values(step, commit=False, **kwargs):
    log_dict = {}
    for key, value in kwargs.items():
        log_dict[key.replace('_', ' ').capitalize()] = value
    wandb.log(log_dict, step=step, commit=commit)


def log_validation(mel_loss, gate_loss, attn_loss, y, y_pred, input_lengths, output_lengths, step, waveglow_path):
    _, mel_outputs, gate_outputs, alignments = y_pred
    mel_targets, gate_targets = y

    # plot alignment, mel target and predicted, gate target and predicted
    times = 3 if alignments.size(0) >= 3 else alignments.size(0)
    idxs = []
    alignmentss, mels, predicteds, gates = [], [], [], []

    waveglow, audios = None, None
    if waveglow_path:
        waveglow = torch.load(waveglow_path)['model']
        waveglow.cuda().eval().half()
        for k in waveglow.convinv:
            k.float()
        audios = []

    for x in range(times):
        # plot alignment, mel target and predicted, gate target and predicted
        idx = random.randint(0, alignments.size(0) - 1)
        while idx in idxs:
            idx = random.randint(0, alignments.size(0) - 1)
        idxs.append(idx)
        lengths = [input_lengths[idx].data.cpu().numpy(), output_lengths[idx].data.cpu().numpy()]
        alignmentss.append(plot_alignment_to_numpy(alignments[idx].data.cpu().numpy().T[:lengths[0], :lengths[1]],
                                                   wandb_im=True))
        mels.append(plot_spectrogram_to_numpy(mel_outputs[idx, :, :lengths[1]].data.cpu().numpy(),
                                              mel_targets[idx, :, :lengths[1]].data.cpu().numpy(),
                                              wandb_im=True))
        gates.append(plot_gate_outputs_to_numpy(gate_targets[idx, :lengths[1]].data.cpu().numpy(),
                                                torch.sigmoid(gate_outputs[idx, :lengths[1]]).data.cpu().numpy(),
                                                wandb_im=True))
        if waveglow is not None:
            with torch.no_grad():
                audio = waveglow.infer(mel_outputs[idx, :, :lengths[1]].unsqueeze(0).cuda().half(), sigma=0.666)
                audios.append(wandb.Audio(audio[0].to(torch.float32).data.cpu().numpy(),
                                          sample_rate=22050)
                              )

    log = {"Validation mel loss": mel_loss, "Validation gate loss": gate_loss, "Validation attention loss": attn_loss,
           "Alignment": alignmentss, 'Mel spectrogram': mels, 'Gate': gates}
    if audios is not None:
        log['Audios'] = audios

    wandb.log(log, step=step)
