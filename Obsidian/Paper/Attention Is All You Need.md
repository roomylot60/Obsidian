[Original Paper Link](https://arxiv.org/abs/1706.03762)

---
## Abstract

Sequence transduction models : Encoder + Decoder (feat. Multi-layer CNN or RNN)
Suggestion : Transformer Architecture with Attention machanism
Parallelizable and less time in training
Machine translation 8 GPUs for 3.5 days of training
28.4 BLEU score on WMT 2014 English-to-German translation task
41.8 BLEU score on WMT 2014 English-to-Franch translation task

---
## 1. Introduction

RNN, LSTM, Gated RNN in sequence modeling
Hidden states to align teh positions of the input and output sequences made of input of present time step and previous hidden state
Efficiency loss : Difficulty of Parallelization caused by inherently sequential nature
[Factorization trick](https://arxiv.org/abs/1703.10722) and [Conditional computation](https://arxiv.org/abs/1701.06538)
Attention machanisms just used in conjunction with a RNN in spite of their usability in modeling of dependencies without regard to their distance in the input or ouput sequences

---
## 2. Background

The goal of reducing sequential computations
CNN base models compute hidden representations in parallel for all input and output positions
Learning dependencies between distant positions is difficult cause of the numerous operations
Self-attention; intra-attention relates different positions of a single sequence

---
## 3. Model Architecture

### 3.1 Encoder and Decoder Stacks

### 3.2 Attention

#### 3.2.1 Scaled Dot-product Attention

#### 3.2.2 Multi-head Attention

#### 3.2.3 Applications of Attention in our Model

### 3.3 Position-wise Feed-Forward Networks

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

