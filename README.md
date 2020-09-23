# GANtron (without wavenet)

PyTorch implementation of GANtron: Emotional Speech Synthesis with Generative Adversarial Networks.
Model based on [Tacotron 2](https://github.com/NVIDIA/tacotron2) 
Natural TTS Synthesis By Conditioning Wavenet On Mel Spectrogram Predictions.

This implementation uses the [LJSpeech dataset](https://keithito.com/LJ-Speech-Dataset/) 
and the [VESUS dataset](https://engineering.jhu.edu/nsa/vesus/). 
You may also need the [CREMA-D dataset](https://github.com/CheyneyComputerScience/CREMA-D) 
and the [RAVDESS dataset](https://zenodo.org/record/1188976).


## Pre-requisites
1. NVIDIA GPU + CUDA cuDNN

## Setup
1. Download and extract the [LJ Speech dataset](https://keithito.com/LJ-Speech-Dataset/) and the [VESUS dataset](https://engineering.jhu.edu/nsa/vesus/).
2. Clone this repo: `git clone https://github.com/Brechard/GANtron`
3. CD into this repo: `cd GANtron`
4. Initialize submodule: `git submodule init; git submodule update`
5. Install PyTorch
6. Install python requirements or build docker image 
    - Install python requirements: `pip install -r requirements.txt`
7. (optional) Download [WaveGlow](https://drive.google.com/open?id=1rpK8CzAAirq9sWZhe9nlfvxMF1dRgFbF) model

## Training
Every configuration training has this line in common:
`python train.py --output_directory=outdir --wavs_path=path_to_LJ_dataset --waveglow_path=waveglow_path`
The extra parameters for the 4 different configurations are:
1. Vanilla GANtron: `--hparams use_labels=False,use_noise=True`
2. Using VESUS with Noise: `--hparams use_labels=False,use_noise=True --vesus_path=vesus_path`
3. Using VESUS, only labels: `--hparams use_labels=True,use_noise=False --vesus_path=vesus_path`
4. Combining the models: `--hparams use_labels=True,use_noise=True --vesus_path=vesus_path`

## Studying each trained model
Several flags are provided in case we want to force the noise, the labels, 
use predefined labels instead of random for each group and using INT values 
for the values in the labels. Set those according to your needs.

`python study_models.py --gantron_path=path_to_ckpt --waveglow_path=waveglow_path --output_path=output_dir ----hparams specific_hparams_from_model`


## Training using a pre-trained model
Training using a pre-trained model can lead to faster convergence  
By default, the dataset dependent text embedding layers are ignored.

## Related repos
[Tacotron 2](https://github.com/NVIDIA/tacotron2) Natural TTS Synthesis By Conditioning Wavenet On Mel Spectrogram Predictions

[WaveGlow](https://github.com/NVIDIA/WaveGlow) Faster than real time Flow-based Generative Network for Speech Synthesis

[nv-wavenet](https://github.com/NVIDIA/nv-wavenet/) Faster than real time WaveNet.

## Acknowledgements
This implementation uses code from the following repos: [Keith
Ito](https://github.com/keithito/tacotron/), [Prem
Seetharaman](https://github.com/pseeth/pytorch-stft), [Rafael Valle](https://github.com/NVIDIA/tacotron2) as described in our code.

We are inspired by [Ryuchi Yamamoto's](https://github.com/r9y9/tacotron_pytorch)
Tacotron PyTorch implementation.

We are thankful to the Tacotron 2 paper authors, specially Jonathan Shen, Yuxuan
Wang and Zongheng Yang.


[WaveGlow]: https://drive.google.com/open?id=1rpK8CzAAirq9sWZhe9nlfvxMF1dRgFbF
[Apex]: https://github.com/nvidia/apex
[AMP]: https://github.com/NVIDIA/apex/tree/master/apex/amp