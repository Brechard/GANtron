import argparse
import ast

from text import symbols


class HParams:
    def __init__(self, hparams_string=None):
        """Create model hyperparameters. Parse nondefault from given string."""
        ################################
        # Experiment Parameters        #
        ################################
        self.epochs = 500
        self.iters_per_checkpoint = 1000
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

        ################################
        # Data Parameters             #
        ################################
        self.load_mel_from_disk = False
        self.training_files = 'filelists/ljs_audio_text_train_filelist.txt'
        self.validation_files = 'filelists/ljs_audio_text_val_filelist.txt'
        self.text_cleaners = ['english_cleaners']

        ################################
        # Audio Parameters             #
        ################################
        self.max_wav_value = 32768.0
        self.sampling_rate = 22050
        self.filter_length = 1024
        self.hop_length = 256
        self.win_length = 1024
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
        self.max_decoder_steps = 1000
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
            for param in hparams_string.split(','):
                key, value = param.split('=')
                self.add_param(key, ast.literal_eval(value))

    def add_param(self, param, value):
        self.__setattr__(param, value)

    def add_params(self, params):
        if type(params) is argparse.Namespace:
            params = params.__dict__

        for param, value in params.items():
            if param == 'hparams':
                continue
            if value is not None:
                self.add_param(param, value)
