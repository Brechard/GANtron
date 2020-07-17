import torch
import torch.nn.functional as F
from torch import nn


class Tacotron2Loss(torch.nn.Module):
    def __init__(self):
        super(Tacotron2Loss, self).__init__()

    def forward(self, model_output, targets, input_lengths, output_lengths):
        mel_target, gate_target = targets[0], targets[1]
        mel_target.requires_grad = False
        gate_target.requires_grad = False
        gate_target = gate_target.view(-1, 1)

        mel_out, mel_out_postnet, gate_out, alignments = model_output
        gate_out = gate_out.view(-1, 1)
        mel_loss = nn.MSELoss()(mel_out, mel_target) + \
                   nn.MSELoss()(mel_out_postnet, mel_target)
        gate_loss = nn.BCEWithLogitsLoss()(gate_out, gate_target)
        attn_loss = attention_loss(alignments, input_lengths, output_lengths)
        return mel_loss, gate_loss, attn_loss


def attention_loss(attention_weights, encoded_length, decoded_length):
    attention_loss = 0
    for batch in range(attention_weights.size(0)):
        mask = attention_mask(encoded_length[batch], decoded_length[batch]).cuda()
        attention_loss += F.binary_cross_entropy(
            attention_weights[batch].T[:encoded_length[batch], :decoded_length[batch]], mask)
    attention_loss /= attention_weights.size(0)
    return attention_loss


def get_sig():
    if int(torch.__version__.replace('.', '')[:3]) > 120:
        return torch.cuda.IntTensor([3])
    else:
        return torch.cuda.FloatTensor([3])


def gaussian(x, center, sig):
    return torch.exp(-torch.pow(x - center, 2.) / (2 * torch.pow(sig, 1)))


def attention_mask(n_chars, n_frames):
    mask = torch.zeros((n_chars, n_frames), device='cuda')
    for n in range(n_chars):
        mask[n] = gaussian(torch.linspace(0, n_frames - 1, n_frames, device='cuda'),
                           n * (n_frames - 1) // (n_chars - 1),
                           get_sig())
    return mask
