## Transformer Model
- 2017년 Google에서 발표한 "Attention is all you need" 논문에서 제시한 모델
- Attention Machanism 만을 사용하여 seq2seq의 구조인 *encoder-decoder*로 구현
	- seq2seq : Encoder, Decoder에서 각각 하나의 RNN이 t 개의 시점을 가지는 구조
	- transformer : Encoder, Decoder 단위(Layer)가 N 개로 구성되는 구조
- RNN을 사용하지 않았음에도 성능적인 우수성
	- Positional information : RNN은 단어의 위치에 따라 단어를 순차적(Sequential) 처리를 하여 단어의 위치 정보를 보유
	- Positional Encoding : Transformer는 단어의 위치 정보(sin, cos)를 각 단어의 Embedding vector에 더하여 Model의 입력으로 사용![Positional Encoding](../Attatched/Pasted%20image%2020240104161019.png)![Embedding vector + PE](../Attatched/Pasted%20image%2020240104162143.png)


### seq2seq Model의 문제
- Input sequence를 하나의 벡터(context vector)로 압축하는 과정에서의 정보 손실
### Hyper-parameter in Transformer
1. d_{model} : Encoder, Decoder, Embedding vector에서의 차원
2. num_layers : Layer(Encoder+Decoder)의 층 수
3. num_heads : Transformer에서 Attention을 사용할 때 분할 및 병렬 수행, 결과값을 통합하는 방식을 사용하는데, 이 때의 병렬 수
4. d_{ff} : Transformer 내부에 존재하는 Feed Forward Neural Network의 크기(이 때의 FFNN의 입출력층의 크기는 d_{model})

## Code
```python
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

class PositionalEncoding(tf.keras.layers.Layer):
	def __init__(self, position, d_model):
		super(PositionalEncoding, self).__init__()
		self.pos_encoding = self.positional_encoding(position, d_model)

	# position encoding에서 사용할 sin, cos의 각도 지정 함수
	def get_angles(self, position, i, d_model):
		angles = 1 / tf.pow(10000, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
		return position * angles
	
	def positional_encoding(self, position, d_model):
		angle_rads = self.get_angles(
			position=tf.range(position, dtype=tf.float32)[:, tf.newaxis],
			i=tf.range(d_model, dtype=tf.float32)[tf.newaxis, :],
			d_model=d_model)
		
		# Even : 2i
		sines = tf.math.sin(angle_rads[:, 0::2])
		
		# Odd : 2i+1
		cosines = tf.math.cos(angle_rads[:, 1::2])
	
		angle_rads = np.zeros(angle_rads.shape) # 홀수, 짝수에 따라 나누어진 각도들에 대한 정보를 통합
		angle_rads[:, 0::2] = sines
		angle_rads[:, 1::2] = cosines
		pos_encoding = tf.constant(angle_rads)
		pos_encoding = pos_encoding[tf.newaxis, ...]
		print(pos_encoding.shape)
		
		return tf.cast(pos_encoding, tf.float32)

	def call(self, inputs):
		return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]
```
[Transformer_Korean_Chatbot](../Attatched/Transformer_Korean_Chatbot.ipynb)

---
## Attention in TM
### Encoder Self-Attention
### Masked Decoder Self-Attention
### Encoder-Decoder Attention
---
## Encoder
### First sublayer
#### Self-Attention
- Attention : Query에 대해서, Key와의 유사도를 mapping된 각각의 Value에 반영, Value의 가중합을 return
```python
Q = Query # t 시점의 decoder cell에서의 hidden state = t 시점의 encoder에서의 context vector
K = Keys # 모든 시점의 encoder cell의 hidden states = dot product의 대상이 되는 set
V = Values # 모든 시점의 encoder cell의 hidden states = weight과 곱해지는 vector의 set
```
- Self-Attention : Q, K, V가 모두 입력 문장의 모든 단어 벡터들을 의미
	- d_{model}의 차원을 갖는 단어 벡터들을 num_heads로 나눈 값을 Q, K, V의 벡터의 차원으로 결정
- Scaled dot-product Attention
- Multi-head Attention : Self Attention을 병렬적으로 사용
- Padding Mask
- Residual connection(잔차 연결)
- Layer Normalization(층 정규화)
### Second sublayer : FFNN
---
## Decoder
### First sublayer
- Self-Attention
- Look-ahead Mask
### Second sublayer
- Encoder-Decoder Attention
---
## Position-wise FFNN
