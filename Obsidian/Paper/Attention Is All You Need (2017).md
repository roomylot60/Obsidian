[Original Paper Link](https://arxiv.org/abs/1706.03762)
[Ref.](https://incredible.ai/nlp/2020/02/29/Transformer/)

---
## Abstract

- Sequence transduction models : Encoder + Decoder (feat. Multi-layer CNN or RNN)
- Suggestion : **Transformer Architecture with Attention mechanism**
	- Parallelizable and less time in training
	- Machine translation 8 GPUs for 3.5 days of training
	- 28.4 BLEU score on WMT 2014 English-to-German translation task
	- 41.8 BLEU score on WMT 2014 English-to-Franch translation task

---
## 1. Introduction

- *Sequence modeling*과 *변환(transduction)* 문제에서 활용도를 넓혀가는 Recurrent 언어 모델과 Encoder-decoder 구조(with RNN, LSTM, Gated RNN)
- 입출력 시퀀스의  문자 위치에 따라 연산이 이루어지는데, 각 시점마다 연산을 위해 문자 위치를 나열할 때, 이전 시점의 은닉상태와 입력으로 구성된 함수로 은닉상태의 시퀀스를 생성해야 한다
- *Hidden states* to align the positions of the input and output sequences made of input of present time step and previous hidden state
- Efficiency loss : *Difficulty of Parallelization* caused by inherently sequential nature
- [Factorization trick](https://arxiv.org/abs/1703.10722) and [Conditional computation](https://arxiv.org/abs/1701.06538)
- *Attention mechanisms* just used in conjunction with a RNN in spite of their usability in modeling of dependencies without regard to their distance in the input or ouput sequences

---
## 2. Background

- The goal of *reducing sequential computations*
- CNN base models compute hidden representations *in parallel* for all input and output positions
- Learning dependencies between distant positions is difficult cause of the *numerous operations*
- Self-attention; intra-attention relates different positions of a *single sequence* and makes this into a representation
- End-to-end memory networks are based on a *recurrent attention mechanism* instead of sequence-aligned recurrence
- **Transformer** is the first transduction mdel relying entirely on self-attention to compute representations of its input and output without using sequence-aligned RNNs or convolution

---
## 3. Model Architecture

![](Attatched/Pasted%20image%2020240415092511.png)

- self-attention and point-wise, fully connected layers
### 3.1 Encoder and Decoder Stacks

- Encoder : Two sub-layers for each layer make output $LayerNorm(x+Sublayer(x)), d_{model}=512$
	- Multi-head self-attention mechanism
	- Position-wise fully connected FFNN
- Decoder : Employ residual connections around each of the sub-layers and insert third sub-layer to perform multi-head attention over the output of the encoder stack
### 3.2 Attention

![](../Attatched/Pasted%20image%2020240415195408.png)

- Attention function : Mapping a query and a set of key-value pairs to an output
	- Additive attention
	- Dot-product; multi-plicative attention
#### 3.2.1 Scaled Dot-Product Attention

- Softmax Function : $\frac{e^{x_{i}}}{\sum_{j=1}^{n}e^{x_{j}}}$
- Scaled Dot-Product Attention : $Attention(Q, K, V) = softmax(\frac{QK^{T}}{\sqrt{d_{k}}})V$
- $Q$ : matrix of queries(dimension $d_{k}$), $K$ : matrix of keys(dimension $d_{k}$), $V$ : matrix of values(dimension $d_{v}$)
- Sclaing with $\frac{1}{\sqrt{d_{k}}}$ : solving gradient vanishing problem
	1. Additive attention : computes the compatitbility function using a FFN with a single hidden layer
	2. Dot-product(multiplicative) attention : much faster and more space-efficient
#### 3.2.2 Multi-Head Attention

- Multihead Attention : $$\begin{aligned}MultiHead(Q, K, V) &= Concat(head_{1}, ..., head_{h})W^{O} \\
where\ head_{i} &= Attentions(QW_{i}^{Q}, KW_{i}^{K}, VW_{i}^{V})\end{aligned}$$

#### 3.2.3 Applications of Attention in our Model

1. **Encoder-decoder Attention** : Queries come from the previous decoder layer, Memory keys and values come from the output of the encoder
2. Encoder : Contains self-attention layers
3. Decoder : Self-attention layers

### 3.3 Position-wise Feed-Forward Networks

- For attention in sub-layers, each of the layers need fully connected FFN
- Two linear transformations with a ReLU activation in between
- $FFN(x) = max(0, xW_{1} + b_{1})W_{2} + b_{2}$
- Dimensionality of input and output is $d_{model} = 512$ and the inner-layer has dimensionality $d_{f \ f} = 2048$
- 

### 3.4 Embeddings and Softmax

---
## 4. Why Self-Attention

---
## 5. Training

### 5.1 Training Data and Batching

### 5.2 Hardware and Schedule

### 5.3 Optimizer

### 5.4 Regularization

---
## 6. Results

### 6.1 Machine Translation

### 6.2 Model Variations
### 6.3 English Constituency Parsing

---
## 7. Conclusion

