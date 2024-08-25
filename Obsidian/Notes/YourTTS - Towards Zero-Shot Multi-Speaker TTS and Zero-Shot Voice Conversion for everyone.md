[Original Paper Link](https://arxiv.org/abs/2112.02418)
[Ref.](https://kimjy99.github.io/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0/yourtts/)

---
## Abstract

- Multi-lingual approach to the task of zero-shot multi-speaker TTS
- VITS model
- VCTK dataset
- Promise the *possibilities* for zero-shot multi-speaker TTS and zero-shot voice conversion systems *in low-resource languages*
- very different voice or record characteristics

## 1. Introduction

- Test-To-Speech; TTS system 
- Previous TTS systems were tailored from a *single speaker's voice*
- Recently *ZS-TTS(Zero-shot multi-speaker TTS)* was proposed with several methods like *DeepVoice3 method*
- **Tacotron2** 
	- was adapted using external speaker embeddings extracted from a trained speaker encoder using a *generalized end-to-end loss; GE2E*
	- *LDE embeddings* to improve similarity and naturalness of speech for unseen speakers
- Gender-dependent model improves the similarity for unseen speakers
- **Attentron** 
	- proposed a *finegrained encoder* with a attention mechanism for extracting detailed styles from various reference samples and a *coarsegrained encoder*
- Contributions of **YourTTS** approach
	- State-of-the-art results in the English Language; 
	- The first work proposing a multilingual approach in the zero-shot multi-speaker TTS scope;
	- Ability to do zero-shot multi-speaker TTS and zero-shot Voice Conversion with promising quality and similarity in a target language using only one speaker in the target language during model training;
	- Require less than 1 minute of speech to fine-tune the model for speakers who have voice/recording characteristics very different from those seen in model training, and still achieve good similarity and quality results.

학습되지 않은 목소리에 대해 유사성이 높고, 자연스러움을 가질 수 있도록 하는 다 언어 목소리 변환 기술을 목표로 여러 메커니즘과 기술을 활용한 ZS-TTS가 등장하고 있음

---
## 2. YourTTS Model

## 3. Experiments

### 3.1. Speaker Encoder

### 3.2. Audio datasets

### 3.3. Experimental setup

## 4. Results and Discussion

### 4.1. VCTK dataset

### 4.2. LibriTTS dataset

### 4.3. Portuguese MLS datset

### 4.4. Speaker Consistency Loss


## 5. Zero-Shot Voice Conversion

### 5.1. Intra-lingual results

### 5.2. Cross-lingual results

## 6. Speaker Adaptation

## 7. Conclusions, limitations and future work

## 8. Acknowledgements
