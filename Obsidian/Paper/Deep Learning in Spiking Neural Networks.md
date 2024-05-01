[Original Paper Link](https://arxiv.org/abs/1804.08150)
https://jinprelude.github.io/posts/SNN-Basic-Tutorial-1-Spiking-Neural-Network%EB%9E%80/

---

## 1. INTRODUCTION

인공 신경망(ANNs)은 일반적으로 이상적인 컴퓨팅 단위를 사용하여 구축됩니다. 
이들은 연속적인 활성화 값을 갖고 있으며 가중치가 적용된 입력을 받습니다. 
이러한 단위는 생물학적 영감으로 '뉴런'이라고 불립니다. 
이러한 (비스파이킹) 뉴런들은 미분 가능한 비선형 활성화 함수를 사용합니다. 
비선형 활성화 함수는 여러 층을 쌓는 것이 의미가 있게 만들어주며, 그 미분 가능성으로 인해 경사 기반의 최적화 방법을 사용하여 훈련할 수 있습니다. 
최근 대량의 레이블이 달린 데이터 집합, 일반 목적의 GPU 컴퓨팅 형태로의 컴퓨팅 파워, 그리고 고급 정규화 방법의 발전으로 인해 이러한 네트워크는 매우 깊어졌으며 (수십 개의 층), 보지 못한 데이터에 대한 일반화 능력이 크게 향상되었습니다. 
이러한 네트워크의 성능에서 큰 발전이 있었습니다.

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