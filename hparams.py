import argparse
import ast

from text import symbols


class HParams:
    def __init__(self, hparams_string=None):
        """Create model hyperparameters. Parse nondefault from given string."""
        ################################
        # Experiment Parameters        #
        ################################
        self.epochs = 200
        self.iters_per_checkpoint = 5000
        self.seed = 1234
        self.dynamic_loss_scaling = True
        self.fp16_run = False
        self.distributed_run = False
        self.dist_backend = "nccl"
        self.dist_url = "tcp://localhost:54321"
        self.cudnn_enabled = True
        self.cudnn_benchmark = False
        self.ignore_layers = ['decoder.attention_rnn.weight_ih',
                              'decoder.attention_layer.memory_layer.linear_layer.weight',
                              'decoder.decoder_rnn.weight_ih', 'decoder.linear_projection.linear_layer.weight',
                              'decoder.gate_layer.linear_layer.weight']
        self.attn_steps = 5000
        self.reduce_lr_steps_every = 5e4
        self.vesus_path = None
        self.speakers_embedding = 64
        self.use_labels = True
        self.use_noise = False
        self.use_intended_labels = True

        ################################
        # Data Parameters             #
        ################################
        self.load_mel_from_disk = False
        self.training_files = ['filelists/ljs_audio_text_train_filelist.txt', 'filelists/vesus_train.txt']
        self.validation_files = ['filelists/ljs_audio_text_val_filelist.txt', 'filelists/vesus_val.txt']
        self.text_cleaners = ['english_cleaners']

        ################################
        # Audio Parameters             #
        ################################
        self.max_wav_value = 32768.0
        self.sampling_rate = 22050
        self.filter_length = 1024
        self.hop_length = 256
        self.win_length = 1024
        self.n_ftt = 1024
        self.n_mel_channels = 80
        self.mel_fmin = 0.0
        self.mel_fmax = 8000.0

        ################################
        # Model Parameters             #
        ################################
        self.n_symbols = len(symbols)
        self.symbols_embedding_dim = 512

        # Encoder parameters
        self.encoder_kernel_size = 5
        self.encoder_n_convolutions = 3
        self.encoder_embedding_dim = 512

        # Decoder parameters
        self.n_frames_per_step = 1  # currently only 1 is supported
        self.decoder_rnn_dim = 1024
        self.prenet_dim = 256
        self.max_decoder_steps = 500
        self.gate_threshold = 0.5
        self.p_attention_dropout = 0.1
        self.p_decoder_dropout = 0.1

        # Attention parameters
        self.attention_rnn_dim = 1024
        self.attention_dim = 128

        # Location Layer parameters
        self.attention_location_n_filters = 32
        self.attention_location_kernel_size = 31

        # Mel-post processing network parameters
        self.postnet_embedding_dim = 512
        self.postnet_kernel_size = 5
        self.postnet_n_convolutions = 5

        # GAN parameters
        self.discriminator_window = 20
        self.discriminator_dim = 512
        self.g_freq = 2
        self.d_freq = 1
        self.clipping_value = 0.001
        self.gradient_penalty_lambda = 0
        self.noise_size = 20
        self.disc_warmp_up = 500
        self.discriminator_type = 'linear'
        self.encoder_emotions = False

        ################################
        # Optimization Hyperparameters #
        ################################
        self.use_saved_learning_rate = False
        self.g_learning_rate = 0.001
        self.d_learning_rate = 0.0007
        self.weight_decay = 1e-6
        self.grad_clip_thresh = 1.0
        self.batch_size = 32
        self.mask_padding = True  # set model's padded outputs to padded values

        if hparams_string:
            self.add_params_string(hparams_string)

    def add_params_string(self, hparams_string):
        for param in hparams_string.split(','):
            key, value = param.split('=')
            if '/' in value:
                self.add_param(key, value)
            else:
                self.add_param(key, ast.literal_eval(value))

    def add_param(self, param, value):
        self.__setattr__(param, value)

    def add_params(self, params):
        if type(params) is str and '=' in params:
            self.add_params_string(params)
            return

        if type(params) is argparse.Namespace:
            params = params.__dict__
        hparams_string = None
        for param, value in params.items():
            if param == 'hparams':
                hparams_string = value
            elif value is not None:
                self.add_param(param, value)

        if hparams_string is not None:
            # HParams passed in the hparams argument has the highest priority.
            self.add_params_string(hparams_string)

