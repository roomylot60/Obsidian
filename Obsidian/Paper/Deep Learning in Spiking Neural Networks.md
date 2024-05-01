[Original Paper Link](https://arxiv.org/abs/1804.08150)
https://jinprelude.github.io/posts/SNN-Basic-Tutorial-1-Spiking-Neural-Network%EB%9E%80/

---

## 1. INTRODUCTION

ANNs는 활성화 값, 가중치가 부과된 입력값의 집합을 연속적으로 연산하는 단위체로 고안되었다.
생물체로부터 영감을 얻어 해당 유닛을 뉴런이라 칭한다.
해당 뉴런들은 미분가능한 비 선형적인 활성화 함수를 사용
비 선형 활성화 함수들은 하나 이상의 계층의 누적을 통해 의미있는 표현을 생성하고 그 산출물들은 학습에서 기울기 기반의 최적화 수단을 사용 가능하도록 한다
거대 라벨링 데이터 집합의 이용가능성에서의 최근의 발전은 GPU 연산의 일반적인 목표와 향상된 정규화 수단들의 형태에서의 연산 능력과 

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