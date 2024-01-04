## Transformer Model
- 2017년 Google에서 발표한 "Attention is all you need" 논문에서 제시한 모델
- Attention Machanism 만을 사용하여 seq2seq의 구조인 *encoder-decoder*로 구현
	- seq2seq : Encoder, Decoder에서 각각 하나의 RNN이 t 개의 시점을 가지는 구조
	- transformer : Encoder, Decoder 단위(Layer)가 N 개로 구성되는 구조
- RNN을 사용하지 않았음에도 성능적인 우수성
	- Positional information : RNN은 단어의 위치에 따라 단어를 순차적(Sequential) 처리를 하여 단어의 위치 정보를 보유
	- Positional Encoding : Transformer는 단어의 위치 정보를 각 단어의 Embedding vector에 더하여 Model의 입력으로 사용![](../Attatched/Pasted%20image%2020240104161019.png)
```python
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
```
[Transformer_Korean_Chatbot](../Attatched/Transformer_Korean_Chatbot.ipynb)

### seq2seq Model의 문제
- Input sequence를 하나의 벡터(context vector)로 압축하는 과정에서의 정보 손실
### Hyper-parameter in Transformer
1. d_{model} : Encoder, Decoder, Embedding vector에서의 차원
2. num_layers : Layer(Encoder+Decoder)의 층 수
3. num_heads : Transformer에서 Attention을 사용할 때 분할 및 병렬 수행, 결과값을 통합하는 방식을 사용하는데, 이 때의 병렬 수
4. d_{ff} : Transformer 내부에 존재하는 Feed Forward Neural Network의 크기(이 때의 FFNN의 입출력층의 크기는 d_{model})