[Original Paper Link](https://www.researchgate.net/publication/13853244_Long_Short-term_Memory)
[Ref.]()

---
## Abstract

---
## 1 INTRODUCTION

- Recurrent network : Feedback connections to store representations of recent input events in form of activations
- Existing short-term memory takes too much time for learning something and do not work well when minimal time lags getting longer between inputs and teacher signals
- **The problem**
	1. Error signals *explosion* in backprop
	2. Error signals *vanishing* in backprop
- **The remedy**
	- **Long Short-Term Memory; LSTM** : Recurrent Network architecture in conjunction with an appropriate gradient-based learning algorithm

---

## 2 PREVIOUS WORK
review previous work

- **Gradient-descent variants** : *Back-Propagation Through Time* and *Real-Time Recurrent Learning*'s  problems
- **Time-delays** : Practical for short time lags; Updates unit activations based on a weighted sum of old activations
- **Time constants** : To deal with long time lags
- Rings' approach : Whenever receive conflicting error signals, add a higher order unit
- Bengio et al.'s approaches
- Kalman filters : To improve recurrent net performance
- **Second order nets** : When using multiplictive units(MUs) to protect error flow from unwanted perturbations
- Simple weight guessing : Faster way to solve many problems of random iniitalizing for all network weights to avoid long time lag problems of gradient-based approaches
- Adaptive sequence chunkers : Holding a capability to bridge arbitrary time lags
---
## 3 CONSTANT ERROR BACKPROP
detailed analysis of vanishing errors in backprop for didactic purposes

### 3.1 EXPONENTIALLY DECAYING ERROR

- Conventional Back-Propagation Through Time

### 3.2 CONSTANT ERROR FLOW : NAIVE APPROACH

---
## 4 LONG SHORT-TERM MEMORY
introduce LSTM

- **Memory cells and gate units** : Differnt type of units convey useful information about the current state of the net
	- Input gate unit : To protect the memory contents stored in linear unit $j$ 
	- Output gate unit : Protects other units from perturbatoin by currently irrelevant memory contents stored in $j$
	- Memory cell : with a fixed self-connection, memory cell is built around a central linear unit

---
## 5 EXPERIMENTS
present numerous experiments and comparisons with competing methods

### 5.1 EXPERIMENT 1 : EMBEDDED REBER GR AMMAR

### 5.2 EXPERIMENT 2 : NOISE-FREE AND NOISY SEQUENCES

### 5.3 EXPERIMENT 3 : NOISE AND SIGNAL ON SAME CHANNEL

### 5.4 EXPERIMENT 4 : ADDING PROBLEM

### 5.5 EXPERIMENT 5 : MULTIPLICATION PROBLEM

### 5.6 EXPERIMENT 6 : TEMPORAL ORDER

### 5.7 EXPERIMENT 7 : SUMMARY OF EXPERIMENTAL CONDITIONS

---
## 6 DISCUSSION
discuss LSTM's limitations and advantages

---
## 7 CONCLUSION

---
## 8 ACKNOWLEDMENTS