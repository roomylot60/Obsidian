[Original Paper Link](https://arxiv.org/abs/1804.08150)
https://jinprelude.github.io/posts/SNN-Basic-Tutorial-1-Spiking-Neural-Network%EB%9E%80/

---

## 1. INTRODUCTION

- 인공 신경망(ANNs)은 일반적으로 연속적인 활성화 값을 갖고 있으며 가중치가 부여된 입력을 받는 이상적인 컴퓨팅 단위를 사용하여 생성
- 이러한 단위는 생물학적 영감에서 비롯되어 '뉴런'이라고 명명됨
- 이러한 (비 스파이킹) 뉴런들은 미분 가능한 비선형 활성화 함수를 사용
- 비선형 활성화 함수는 하나 이상의 계층을 쌓는 것이 의미가 있도록 하며, 도함수의 존재;미분 가능성은 경사 기반의 최적화 수단들을 훈련에 사용할 수 있도록 함
- 최근 대량의 레이블이 달린 데이터 집합, 일반 목적의 GPU 연산 형태로의 연산 능력, 그리고 고급 정규화 방법의 발전으로 인해, 미 규정 데이터에 대한 뛰어난 일반화 능력과 함께 해당 네트워크는 매우 깊어졌으며 (수십 개의 층으로 구성), 네트워크의 성능에서 큰 발전이 이루어짐

- 눈에 띄는 역사적인 쾌거로는 2012년의 ILSVRC 이미지 분류 챌린지에서의 AlexNet의 성공이 있음
- 총 6천만 개의 학습 파라미터를 사용하는 선단 간(end-to-end) 학습을 통한 8개의 연속적인 계층으로 구성되어 있어 AlexNet은 심층 신경망(DNN)으로 알려져 있음
- 최근의 DNNs의 연구 [`[3]`](Y. LeCun, Y. Bengio, and G. Hinton, “Deep learning,” Nature, vol. 521, no. 7553, pp. 436–444, 2015.), [`[4]`](J. Schmidhuber, “Deep learning in neural networks: An overview,” Neural Networks, vol. 61, pp. 85–117, 2015.)
- DNNs 많은 이미지 인식, 객체 탐지, 음성 인식, 생의학, 연속 데이터 작업을 비롯한 애플리케이션에서 눈에 띄게 성공을 거둠
- 이러한 AI에서의 최근의 발전은 다른 공학 애플리케이션 개발과 생물적으로 뇌의 기능을 이해하는 데에 있어서 새로운 활로 개척

- 비록 DNNs가 역사적으로 뇌로 부터 비롯되었다고는 하나, 뇌와 구조적, 신경 연산, 학습법 등에서 기본적인 차이를 보임
- 가장 주요한 차이점은 유닛간의 정보 전파의 방식
- 
---

## 2. Spiking Neural Network : A biologically inspired approach to information precessing

---

## 3. Deep Learning in SNNs

### A. Deep, Fully Connected SNNs

### B. Spiking CNNs

### C. Spiking Deep Belief Networks

### D. Recurrent SNNs

### E. Performance Comparisions of Contemporary Models
---

## 4. Summary

---
### SNN

- SNN : 기존 DL Neworks(MLP, RNN, CNN etc..)가 Tensor, Float을 주고받는 것에 비해 특정 뉴런에서 특정 시간에 Spike 발생 여부에 대한 이산적인 정보를 주고 받는 신경망
- Spike의 누적으로 Action potential(활동 전위)가 임계점을 넘으면 Spike를 출력
- Analogue 방식으로 구현하여 DNN에 비해 낮은 전력 소모로 기동 가능
- 형상학적인(topologically) DNN과 달리, 실제 생물의 신경망에서의 작동 방식을 채용하여 Bio-Plausible : Hodgkin-Huxley model
- 효과적인 학습방법의 부재
---
## Ref.
https://www.eenewseurope.com/en/eta-adds-spiking-neural-network-support-to-mcu/
https://jinprelude.github.io/posts/SNN-Basic-Tutorial-2-SNN%EC%9D%84-%EC%9C%84%ED%95%9C-%EA%B8%B0%EC%B4%88-%EB%87%8C%EA%B3%BC%ED%95%99/