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
## Tacotron2
from hparams import defaults # hyperparameter 설정 : dictinoary 형태
from model import Tacotron2 # Pytorch 기반
from layers import TacotronSTFT, STFT # STFT(Short Time Fourier Transform)
from audio_processing import griffin_lim
from tacotron2.train import load_model

## WaveGlow
from waveglow.glow import WaveGlow
from denoiser import Denoiser

## Common
from text import text_to_sequence
from scipy.io.wavfile import write
import IPython.display as ipd
import json
import sys
from tqdm.notebook import tqdm

## WaveGlow 프로젝트 위치 설정
sys.path.append('waveglow/')

## Tacontron2 프로젝트 위치 설정
sys.path.append('tacotron2/')
```
