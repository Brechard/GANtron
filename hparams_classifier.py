import argparse
import ast


# TODO: make abstract HParams class and make this as child of it

class HParams:
    def __init__(self, hparams_string=None):
        """Create model hyperparameters. Parse nondefault from given string."""
        ################################
        # Experiment Parameters        #
        ################################
        self.epochs = 100
        self.precision = 32
        self.cudnn_enabled = True
        self.cudnn_benchmark = False
        self.use_labels = 'intended'
        self.model_version = '0.6'
        # v0.6: Introduced the ability to train the model next to the augmented data from GANtron

        ################################
        # Data Parameters             #
        ################################
        self.training_files = ['filelists/vesus_train.txt', 'filelists/cremad_train.txt', 'filelists/ravdess_train.txt']
        self.validation_files = ['filelists/vesus_val.txt', 'filelists/cremad_val.txt', 'filelists/ravdess_val.txt']
        self.test_files = ['filelists/vesus_test.txt', 'filelists/cremad_test.txt', 'filelists/ravdess_test.txt']
        self.n_emotions = 5

        ################################
        # Audio Parameters             #
        ################################
        self.sampling_rate = 22050
        self.n_ftt = 1024
        self.hop_length = 256
        self.n_mel_channels = 80
        self.mel_offset = 0

        ################################
        # Model Parameters             #
        ################################
        self.linear_model = True
        self.model_size = 256
        self.n_frames = 80

        ################################
        # Optimization Hyperparameters #
        ################################
        self.lr = 0.001
        self.weight_decay = 1e-6
        self.batch_size = 8
        self.max_noise = 5

        if hparams_string:
            self.add_params_string(hparams_string)

    def add_params_string(self, hparams_string):
        for param in hparams_string.split(','):
            key, value = param.split('=')
            if '/' in value:
                self.add_param(key, value)
            else:
                try:
                    self.add_param(key, ast.literal_eval(value))
                except:
                    self.add_param(key, value)

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
