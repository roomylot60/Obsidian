## Tacotron2 & WaveGlow

### Tacotron2
#### [Tacotron2-Wavenet-Korean-TTS(Pytorch)](https://github.com/hccho2/Tacotron2-Wavenet-Korean-TTS) 
- Pytorch 기반으로 생성된 한글 지원 tacotron2 모델
- 한글 자모에 대한 전처리 함수 지원
- Vocoder로 Wavenet을 구현
```python
"""
inference.ipynb 해석
"""
# 기본 라이브러리 Import
import sys
import numpy as np
import torch
import os
import argparse

# 프로젝트 라이브러리 import
from hparams import defaults # hyperparameter 설정 : dictinoary 형태
from model import Tacotron2
from layers import TacotronSTFT, STFT
from audio_processing import griffin_lim
from tacotron2.train import load_model
from text import text_to_sequence
from scipy.io.wavfile import write
import IPython.display as ipd
import json
import sys
from waveglow.glow import WaveGlow
from denoiser import Denoiser
from tqdm.notebook import tqdm

## WaveGlow 프로젝트 위치 설정
sys.path.append('waveglow/')

## Tacontron2 프로젝트 위치 설정
sys.path.append('tacotron2/')
```

```python
class WaveGlow(torch.nn.Module):
    def __init__(self, n_mel_channels, n_flows, n_group, n_early_every, n_early_size, WN_config):
        super(WaveGlow, self).__init__()
        self.upsample = torch.nn.ConvTranspose1d(n_mel_channels, n_mel_channels, 1024, stride=256)
        assert(n_group % 2 == 0)
        self.n_flows = n_flows
        self.n_group = n_group
        self.n_early_every = n_early_every
        self.n_early_size = n_early_size
        self.WN = torch.nn.ModuleList()
        self.convinv = torch.nn.ModuleList()
        n_half = int(n_group/2)
        # Set up layers with the right sizes based on how many dimensions
        # have been output already
        n_remaining_channels = n_group
        for k in range(n_flows):
            if k % self.n_early_every == 0 and k > 0:
                n_half = n_half - int(self.n_early_size/2)
                n_remaining_channels = n_remaining_channels - self.n_early_size
            self.convinv.append(Invertible1x1Conv(n_remaining_channels))
            self.WN.append(WN(n_half, n_mel_channels*n_group, **WN_config))
        self.n_remaining_channels = n_remaining_channels  # Useful during inference
        
    def forward(self, forward_input):
        """
        forward_input[0] = mel_spectrogram:  batch x n_mel_channels x frames
        forward_input[1] = audio: batch x time
        """
        spect, audio = forward_input

  

        #  Upsample spectrogram to size of audio

        spect = self.upsample(spect)

        assert(spect.size(2) >= audio.size(1))

        if spect.size(2) > audio.size(1):

            spect = spect[:, :, :audio.size(1)]

  

        spect = spect.unfold(2, self.n_group, self.n_group).permute(0, 2, 1, 3)

        spect = spect.contiguous().view(spect.size(0), spect.size(1), -1).permute(0, 2, 1)

  

        audio = audio.unfold(1, self.n_group, self.n_group).permute(0, 2, 1)

        output_audio = []

        log_s_list = []

        log_det_W_list = []

  

        for k in range(self.n_flows):

            if k % self.n_early_every == 0 and k > 0:

                output_audio.append(audio[:,:self.n_early_size,:])

                audio = audio[:,self.n_early_size:,:]

  

            audio, log_det_W = self.convinv[k](audio)

            log_det_W_list.append(log_det_W)

  

            n_half = int(audio.size(1)/2)

            audio_0 = audio[:,:n_half,:]

            audio_1 = audio[:,n_half:,:]

  

            output = self.WN[k]((audio_0, spect))

            log_s = output[:, n_half:, :]

            b = output[:, :n_half, :]

            audio_1 = torch.exp(log_s)*audio_1 + b

            log_s_list.append(log_s)

  

            audio = torch.cat([audio_0, audio_1],1)

  

        output_audio.append(audio)

        return torch.cat(output_audio,1), log_s_list, log_det_W_list

  

    def infer(self, spect, sigma=1.0):

        spect = self.upsample(spect)

        # trim conv artifacts. maybe pad spec to kernel multiple

        time_cutoff = self.upsample.kernel_size[0] - self.upsample.stride[0]

        spect = spect[:, :, :-time_cutoff]

  

        spect = spect.unfold(2, self.n_group, self.n_group).permute(0, 2, 1, 3)

        spect = spect.contiguous().view(spect.size(0), spect.size(1), -1).permute(0, 2, 1)

  

        if spect.type() == 'torch.cuda.HalfTensor':

            audio = torch.cuda.HalfTensor(spect.size(0),

                                          self.n_remaining_channels,

                                          spect.size(2)).normal_()

        else:

            audio = torch.cuda.FloatTensor(spect.size(0),

                                           self.n_remaining_channels,

                                           spect.size(2)).normal_()

  

        audio = torch.autograd.Variable(sigma*audio)

  

        for k in reversed(range(self.n_flows)):

            n_half = int(audio.size(1)/2)

            audio_0 = audio[:,:n_half,:]

            audio_1 = audio[:,n_half:,:]

  

            output = self.WN[k]((audio_0, spect))

  

            s = output[:, n_half:, :]

            b = output[:, :n_half, :]

            audio_1 = (audio_1 - b)/torch.exp(s)

            audio = torch.cat([audio_0, audio_1],1)

  

            audio = self.convinv[k](audio, reverse=True)

  

            if k % self.n_early_every == 0 and k > 0:

                if spect.type() == 'torch.cuda.HalfTensor':

                    z = torch.cuda.HalfTensor(spect.size(0), self.n_early_size, spect.size(2)).normal_()

                else:

                    z = torch.cuda.FloatTensor(spect.size(0), self.n_early_size, spect.size(2)).normal_()

                audio = torch.cat((sigma*z, audio),1)

  

        audio = audio.permute(0,2,1).contiguous().view(audio.size(0), -1).data

        return audio

  

    @staticmethod

    def remove_weightnorm(model):

        waveglow = model

        for WN in waveglow.WN:

            WN.start = torch.nn.utils.remove_weight_norm(WN.start)

            WN.in_layers = remove(WN.in_layers)

            WN.cond_layer = torch.nn.utils.remove_weight_norm(WN.cond_layer)

            WN.res_skip_layers = remove(WN.res_skip_layers)

        return waveglow
```