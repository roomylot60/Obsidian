## Paper
[Connectionist Temporal Classification: Labelling Unsegmented Sequence Data with Recurrent Neural Networks](https://www.cs.toronto.edu/~graves/icml_2006.pdf)
#### Abstract
- 실제 sequence 학습에서 사용되는 데이터에는 noise가 많거나 unsegmented 데이터를 사용
- Sequence 학습에서 강점을 보이는 RNN 모델을 사용함에 있어서 데이터의 정제가 필요하나, 이러한 부분의 제약이 많음
- CTC라는 수단을 제안하여 이러한 제약을 해소하고자 함

#### Introduction
- Labelling unsegmented sequence data 작업을 실시간으로 하는 것은 어려움
- Hidden Markov Model; HMM, Conditional Random Fields; CRFs와 같은 Graphical Model들을 활용하는 데에는 이러한 사전 작업이 필요