import matplotlib
# matplotlib.use("Agg")
import matplotlib.pylab as plt
import numpy as np
import wandb

def save_figure_to_numpy(fig):
    # save it to a numpy array.
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return data


def plot_alignment_to_numpy(alignment, info=None, wandb_im=False):
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(alignment, aspect='auto', origin='lower',
                   interpolation='none')
    fig.colorbar(im, ax=ax)
    xlabel = 'Decoder timestep'
    if info is not None:
        xlabel += '\n\n' + info
    plt.xlabel(xlabel)
    plt.ylabel('Encoder timestep')
    plt.tight_layout()

    fig.canvas.draw()
    data = save_figure_to_numpy(fig)
    plt.show()
    plt.close()
    if wandb_im:
        return wandb.Image(data)
    return data


def plot_spectrogram_to_numpy(pred_mel, ground_truth, wandb_im=False):
    fig, (ax1, ax2) = plt.subplots(2, 1)
    im = ax1.imshow(pred_mel, origin="lower")
    ax1.set_title('Generated mel spectrogram')

    im = ax2.imshow(ground_truth, origin="lower")
    ax2.set_title('Ground truth mel spectrogram')

    # Add only one colorbar since they share the y axis
    fig.colorbar(im, ax=[ax1, ax2])
    plt.xlabel("Frames")
    plt.ylabel("Channels")
    # plt.tight_layout()

    fig.canvas.draw()
    data = save_figure_to_numpy(fig)
    plt.show()
    plt.close()
    if wandb_im:
        return wandb.Image(data)
    return data


def plot_gate_outputs_to_numpy(gate_targets, gate_outputs, wandb_im=False):
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.scatter(range(len(gate_targets)), gate_targets, alpha=0.5,
               color='green', marker='+', s=1, label='target')
    ax.scatter(range(len(gate_outputs)), gate_outputs, alpha=0.5,
               color='red', marker='.', s=1, label='predicted')

    plt.xlabel("Frames (Green target, Red predicted)")
    plt.ylabel("Gate State")
    plt.tight_layout()

    fig.canvas.draw()
    data = save_figure_to_numpy(fig)
    plt.show()
    plt.close()
    if wandb_im:
        return wandb.Image(data)
    return data
