import random

import torch
from torch.utils.tensorboard import SummaryWriter

from plotting_utils import plot_alignment_to_numpy, plot_spectrogram_to_numpy
from plotting_utils import plot_gate_outputs_to_numpy


class Tacotron2Logger(SummaryWriter):
    def __init__(self, logdir):
        super(Tacotron2Logger, self).__init__(logdir)

    def log_generator_training(self, total_loss, adv_loss, taco_loss, grad_norm, learning_rate, duration, iteration):
        self.add_scalar("Generator loss", total_loss, iteration)
        self.add_scalar("Tacotron loss", taco_loss, iteration)
        self.add_scalar("Adversarial loss", adv_loss, iteration)
        self.add_scalar("Grad norm", grad_norm, iteration)
        self.add_scalar("Generator learning rate", learning_rate, iteration)
        self.add_scalar("Generation duration", duration, iteration)

    def log_discriminator_training(self, disc_loss, real_loss, fake_loss, learning_rate, duration, iteration):
        self.add_scalar("Discriminator loss", disc_loss, iteration)
        self.add_scalar("Real loss", real_loss, iteration)
        self.add_scalar("Fake loss", fake_loss, iteration)
        self.add_scalar("Discriminator learning rate", learning_rate, iteration)
        self.add_scalar("Discriminator duration", duration, iteration)

    def log_validation(self, reduced_loss, model, y, y_pred, iteration):
        self.add_scalar("validation.loss", reduced_loss, iteration)
        _, mel_outputs, gate_outputs, alignments = y_pred
        mel_targets, gate_targets = y

        # plot distribution of parameters
        for tag, value in model.named_parameters():
            tag = tag.replace('.', '/')
            self.add_histogram(tag, value.data.cpu().numpy(), iteration)

        # plot alignment, mel target and predicted, gate target and predicted
        idx = random.randint(0, alignments.size(0) - 1)
        self.add_image(
            "alignment",
            plot_alignment_to_numpy(alignments[idx].data.cpu().numpy().T),
            iteration, dataformats='HWC')
        self.add_image(
            "mel_target",
            plot_spectrogram_to_numpy(mel_targets[idx].data.cpu().numpy()),
            iteration, dataformats='HWC')
        self.add_image(
            "mel_predicted",
            plot_spectrogram_to_numpy(mel_outputs[idx].data.cpu().numpy()),
            iteration, dataformats='HWC')
        self.add_image(
            "gate",
            plot_gate_outputs_to_numpy(
                gate_targets[idx].data.cpu().numpy(),
                torch.sigmoid(gate_outputs[idx]).data.cpu().numpy()),
            iteration, dataformats='HWC')
