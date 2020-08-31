import argparse
import ast

from text import symbols


class HParams:
    def __init__(self, hparams_string=None):
        """Create model hyperparameters. Parse nondefault from given string."""
        ################################
        # Experiment Parameters        #
        ################################
        self.epochs = 300
        self.precision = 16
        self.cudnn_enabled = True
        self.cudnn_benchmark = False
        self.use_intended_labels = True

        ################################
        # Data Parameters             #
        ################################
        self.training_files = 'filelists/vesus_train.txt'
        self.validation_files = 'filelists/vesus_val.txt'
        self.test_files = 'filelists/vesus_test.txt'
        self.n_emotions = 5

        ################################
        # Audio Parameters             #
        ################################
        self.sampling_rate = 22050
        self.n_ftt = 1024
        self.hop_length = 256
        self.n_mel_channels = 80
        self.mel_offset = 20

        ################################
        # Model Parameters             #
        ################################
        self.linear_model = True
        self.model_size = 512
        self.n_frames = 40

        ################################
        # Optimization Hyperparameters #
        ################################
        self.lr = 0.001
        self.weight_decay = 1e-6
        self.batch_size = 32

        if hparams_string:
            for param in hparams_string.split(','):
                key, value = param.split('=')
                if '/' in value:
                    self.add_param(key, value)
                else:
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
