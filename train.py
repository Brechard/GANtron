import argparse
import math
import os
import time

import torch
import torch.distributed as dist
from numpy import finfo
from torch.autograd import Variable
from torch.autograd import grad as torch_grad
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from data_utils import TextMelLoader, TextMelCollate
from distributed import apply_gradient_allreduce
from hparams import create_hparams
from logger import Tacotron2Logger
from loss_function import Tacotron2Loss
from model import Tacotron2, Discriminator
from utils import get_mask_from_lengths


def round_(tensor, decimals):
    return str(tensor.cpu().detach().numpy().round(decimals))


def gradient_penalty(discriminator, real_data, generated_data, real_lengths, generated_lengths):
    batch_size = real_data.size()[0]

    # Calculate interpolation
    alpha = torch.rand(batch_size, 1, 1).cuda()
    if real_data.size(2) < generated_data.size(2):
        alpha = alpha.expand_as(real_data)
        interpolated = alpha * real_data.data + (1 - alpha) * generated_data.data[:, :, :real_data.size(2)]
        mask = get_mask_from_lengths(real_lengths).unsqueeze(1)
    else:
        alpha = alpha.expand_as(generated_data)
        interpolated = alpha * real_data.data[:, :, :generated_data.size(2)] + (1 - alpha) * generated_data.data
        mask = get_mask_from_lengths(generated_lengths).unsqueeze(1)

    mask = mask.expand_as(interpolated)
    interpolated = interpolated.masked_fill_(mask == False, 0)

    interpolated = Variable(interpolated, requires_grad=True).cuda()

    # Calculate probability of interpolated examples
    prob_interpolated = discriminator(interpolated)

    # Calculate gradients of probabilities with respect to examples
    gradients = torch_grad(outputs=prob_interpolated, inputs=interpolated,
                           grad_outputs=torch.ones(prob_interpolated.size()).cuda(),
                           create_graph=True, retain_graph=True)[0]

    gradients = gradients.masked_fill_(mask == False, 0)
    # Gradients have shape (batch_size, num_channels, img_width, img_height),
    # so flatten to easily take norm per example in batch
    gradients = gradients.reshape(batch_size, -1)

    # Derivatives of the gradient close to 0 can cause problems because of
    # the square root, so manually calculate norm and add epsilon
    gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

    # Return gradient penalty
    return ((gradients_norm - 1) ** 2).mean()


def reduce_tensor(tensor, n_gpus):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.reduce_op.SUM)
    rt /= n_gpus
    return rt


def init_distributed(hparams, n_gpus, rank, group_name):
    assert torch.cuda.is_available(), "Distributed mode requires CUDA."
    print("Initializing Distributed")

    # Set cuda device so everything is done on the right GPU.
    torch.cuda.set_device(rank % torch.cuda.device_count())

    # Initialize distributed communication
    dist.init_process_group(
        backend=hparams.dist_backend, init_method=hparams.dist_url,
        world_size=n_gpus, rank=rank, group_name=group_name)

    print("Done initializing distributed")


def prepare_dataloaders(hparams):
    # Get data, data loaders and collate function ready
    trainset = TextMelLoader(hparams.training_files, hparams)
    valset = TextMelLoader(hparams.validation_files, hparams)
    collate_fn = TextMelCollate(hparams.n_frames_per_step)

    if hparams.distributed_run:
        train_sampler = DistributedSampler(trainset)
        shuffle = False
    else:
        train_sampler = None
        shuffle = True

    train_loader = DataLoader(trainset, num_workers=1, shuffle=shuffle,
                              sampler=train_sampler,
                              batch_size=hparams.batch_size, pin_memory=False,
                              drop_last=True, collate_fn=collate_fn)
    return train_loader, valset, collate_fn


def prepare_directories_and_logger(output_directory, log_directory, rank):
    if rank == 0:
        if not os.path.isdir(output_directory):
            os.makedirs(output_directory)
            os.chmod(output_directory, 0o775)
        logger = Tacotron2Logger(os.path.join(output_directory, log_directory))
    else:
        logger = None
    return logger


def load_model(hparams):
    generator = Tacotron2(hparams).cuda()
    disciminator = Discriminator(hparams).cuda()
    if hparams.fp16_run:
        generator.decoder.attention_layer.score_mask_value = finfo('float16').min

    if hparams.distributed_run:
        generator = apply_gradient_allreduce(generator)
        disciminator = apply_gradient_allreduce(disciminator)

    return generator, disciminator


def warm_start_model(checkpoint_path, model, ignore_layers):
    assert os.path.isfile(checkpoint_path)
    print("Warm starting model from checkpoint '{}'".format(checkpoint_path))
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    model_dict = checkpoint_dict['state_dict']
    if len(ignore_layers) > 0:
        model_dict = {k: v for k, v in model_dict.items()
                      if k not in ignore_layers}
        dummy_dict = model.state_dict()
        dummy_dict.update(model_dict)
        model_dict = dummy_dict
    model.load_state_dict(model_dict)
    return model


def load_checkpoint(checkpoint_path, model, optimizer):
    assert os.path.isfile(checkpoint_path)
    print("Loading checkpoint '{}'".format(checkpoint_path))
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint_dict['state_dict'])
    optimizer.load_state_dict(checkpoint_dict['optimizer'])
    learning_rate = checkpoint_dict['learning_rate']
    iteration = checkpoint_dict['iteration']
    print("Loaded checkpoint '{}' from iteration {}".format(
        checkpoint_path, iteration))
    return model, optimizer, learning_rate, iteration


def save_checkpoint(model, optimizer, learning_rate, iteration, filepath):
    print("Saving model and optimizer state at iteration {} to {}".format(
        iteration, filepath))
    torch.save({'iteration': iteration,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'learning_rate': learning_rate}, filepath)


def validate(model, criterion, valset, iteration, batch_size, n_gpus,
             collate_fn, logger, distributed_run, rank):
    """Handles all the validation scoring and printing"""
    model.eval()
    with torch.no_grad():
        val_sampler = DistributedSampler(valset) if distributed_run else None
        val_loader = DataLoader(valset, sampler=val_sampler, num_workers=1,
                                shuffle=False, batch_size=batch_size,
                                pin_memory=False, collate_fn=collate_fn)

        val_loss = 0.0
        for i, batch in enumerate(val_loader):
            x, y = model.parse_batch(batch)
            y_pred = model(x)
            loss = criterion(y_pred, y)
            if distributed_run:
                reduced_val_loss = reduce_tensor(loss.data, n_gpus).item()
            else:
                reduced_val_loss = loss.item()
            val_loss += reduced_val_loss
        val_loss = val_loss / (i + 1)

    model.train()
    if rank == 0:
        print("Validation loss {}: {:9f}  ".format(iteration, val_loss))
        logger.log_validation(val_loss, model, y, y_pred, iteration)


def train(output_directory, log_directory, checkpoint_path, warm_start, n_gpus,
          rank, group_name, hparams):
    """Training and validation logging results to tensorboard and stdout

    Params
    ------
    output_directory (string): directory to save checkpoints
    log_directory (string) directory to save tensorboard logs
    checkpoint_path(string): checkpoint path
    n_gpus (int): number of gpus
    rank (int): rank of current gpu
    hparams (object): comma separated list of "name=value" pairs.
    """
    if hparams.distributed_run:
        init_distributed(hparams, n_gpus, rank, group_name)

    torch.manual_seed(hparams.seed)
    torch.cuda.manual_seed(hparams.seed)

    generator, discriminator = load_model(hparams)
    g_learning_rate = hparams.g_learning_rate
    d_learning_rate = hparams.d_learning_rate
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=g_learning_rate, weight_decay=hparams.weight_decay)
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=d_learning_rate, weight_decay=hparams.weight_decay)

    if hparams.fp16_run:
        from apex import amp
        generator, g_optimizer = amp.initialize(generator, g_optimizer, opt_level='O2')
        discriminator, d_optimizer = amp.initialize(discriminator, d_optimizer, opt_level='O2')

    if hparams.distributed_run:
        generator = apply_gradient_allreduce(generator)
        discriminator = apply_gradient_allreduce(discriminator)

    criterion = Tacotron2Loss()

    logger = prepare_directories_and_logger(output_directory, log_directory, rank)

    train_loader, valset, collate_fn = prepare_dataloaders(hparams)

    # Load checkpoint if one exists
    iteration = 0
    epoch_offset = 0
    if checkpoint_path is not None:
        if warm_start:
            generator = warm_start_model(checkpoint_path, generator, hparams.ignore_layers)
        else:
            generator, optimizer, _g_learning_rate, iteration = load_checkpoint(
                checkpoint_path, generator, g_optimizer)
            if hparams.use_saved_learning_rate:
                g_learning_rate = _g_learning_rate
            iteration += 1  # next iteration is iteration + 1
            epoch_offset = max(0, int(iteration / len(train_loader)))

    generator.train()
    discriminator.train()
    is_overflow = False
    gen_times, disc_times = 1, 0
    # ================ MAIN TRAINING LOOP! ===================
    for epoch in range(epoch_offset, hparams.epochs):
        print("Epoch: {}".format(epoch))
        for i, batch in enumerate(train_loader):
            start = time.perf_counter()
            if disc_times > 0:
                """ Train Discriminator """
                discriminator.zero_grad()
                x, y = generator.parse_batch(batch)
                real_mel, output_lengths = x[2], x[-1]
                # how well can it label as real?
                real_loss = real * discriminator.adversarial_loss(real_mel, output_lengths)

                # how well can it label as fake?
                fake_loss = fake * discriminator.adversarial_loss(generated_mel.detach(), generated_output_lengths)

                # discriminator loss is the average of these
                discriminator_loss = (real_loss + fake_loss) / 2
                extra_log = ''
                if hparams.clipping_value > 0:
                    torch.nn.utils.clip_grad_norm(discriminator.parameters(), hparams['clipping_value'])
                elif hparams.gradient_penalty_lambda > 0:
                    gp = gradient_penalty(discriminator, real_mel, generated_mel.detach(), output_lengths,
                                          generated_output_lengths)
                    extra_log = f' GP {round_(gp, 3)} '
                    discriminator_loss += hparams.gradient_penalty_lambda * gp

                if hparams.distributed_run:
                    reduced_loss = reduce_tensor(discriminator_loss.data, n_gpus).item()
                else:
                    reduced_loss = discriminator_loss.item()

                discriminator_loss.backward()
                duration = time.perf_counter() - start
                print(f"{iteration} Discriminator loss {round(reduced_loss, 6)} "
                      f"real loss {round_(real_loss, 6)} fake loss {round_(fake_loss, 6)} "
                      f"{extra_log}{duration:.2f}s/it")

                logger.log_discriminator_training(reduced_loss, real_loss, fake_loss, d_learning_rate, duration,
                                                  iteration)

                disc_times += 1
                if disc_times > hparams.d_freq:
                    disc_times = 0
                    gen_times = 1
            else:
                """ Train Generator """
                for param_group in g_optimizer.param_groups:
                    param_group['lr'] = g_learning_rate

                generator.zero_grad()
                x, y = generator.parse_batch(batch)
                y_pred = generator(x)
                generated_mel = y_pred[1]
                generated_output_lengths = x[-1]

                taco_loss = criterion(y_pred, y)
                adv_loss = real * discriminator.adversarial_loss(generated_mel, generated_output_lengths)
                total_loss = taco_loss + adv_loss

                if hparams.distributed_run:
                    reduced_loss = reduce_tensor(total_loss.data, n_gpus).item()
                else:
                    reduced_loss = total_loss.item()

                if hparams.fp16_run:
                    with amp.scale_loss(total_loss, g_optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    total_loss.backward()

                if hparams.fp16_run:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        amp.master_params(g_optimizer), hparams.grad_clip_thresh)
                    is_overflow = math.isnan(grad_norm)
                else:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        generator.parameters(), hparams.grad_clip_thresh)
                g_optimizer.step()
                if not is_overflow and rank == 0:
                    duration = time.perf_counter() - start
                    print(f"{iteration} Generator loss {round_(total_loss, 6)} "
                          f"Adv loss {round_(adv_loss, 6)} Taco loss {round_(taco_loss, 6)} "
                          f"Grad Norm {round_(grad_norm, 6)} {duration:.2f}s/it")

                    logger.log_generator_training(total_loss, adv_loss, taco_loss, grad_norm, g_learning_rate, duration,
                                                  iteration)

                gen_times += 1
                if gen_times > hparams.g_freq:
                    gen_times = 0
                    disc_times = 1

            iteration += 1
            if not is_overflow and (iteration % hparams.iters_per_checkpoint == 0):
                validate(generator, criterion, valset, iteration,
                         hparams.batch_size, n_gpus, collate_fn, logger,
                         hparams.distributed_run, rank)
                if rank == 0:
                    checkpoint_path = os.path.join(
                        output_directory, "checkpoint_{}".format(iteration))
                    save_checkpoint(generator, g_optimizer, g_learning_rate, iteration,
                                    checkpoint_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output_directory', type=str,
                        help='directory to save checkpoints')
    parser.add_argument('-l', '--log_directory', type=str,
                        help='directory to save tensorboard logs')
    parser.add_argument('-c', '--checkpoint_path', type=str, default=None,
                        required=False, help='checkpoint path')
    parser.add_argument('--warm_start', action='store_true',
                        help='load model weights only, ignore specified layers')
    parser.add_argument('--n_gpus', type=int, default=1,
                        required=False, help='number of gpus')
    parser.add_argument('--rank', type=int, default=0,
                        required=False, help='rank of current gpu')
    parser.add_argument('--group_name', type=str, default='group_name',
                        required=False, help='Distributed group name')
    parser.add_argument('--hparams', type=str,
                        required=False, help='comma separated name=value pairs')

    args = parser.parse_args()
    hparams = create_hparams(args.hparams)
    real = 1
    fake = -1

    torch.backends.cudnn.enabled = hparams.cudnn_enabled
    torch.backends.cudnn.benchmark = hparams.cudnn_benchmark

    print("FP16 Run:", hparams.fp16_run)
    print("Dynamic Loss Scaling:", hparams.dynamic_loss_scaling)
    print("Distributed Run:", hparams.distributed_run)
    print("cuDNN Enabled:", hparams.cudnn_enabled)
    print("cuDNN Benchmark:", hparams.cudnn_benchmark)

    train(args.output_directory, args.log_directory, args.checkpoint_path,
          args.warm_start, args.n_gpus, args.rank, args.group_name, hparams)
