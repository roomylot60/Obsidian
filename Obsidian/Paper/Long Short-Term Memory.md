[Original Paper Link](https://www.researchgate.net/publication/13853244_Long_Short-term_Memory)
[Ref.]()

---
## Abstract

---
## 1 INTRODUCTION

- Recurrent network : Feedback connetcions to store representations of recent input events in form of activations
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
- **Time-delays** : 
- Time constants
- Rings' approach
- Bengio et al.'s approaches
- Kalman filters
- Second order nets
- Simple weight guessing
- Adaptive sequence chunkers
---
## 3 CONSTANT ERROR BACKPROP
detailed analysis of vanishing errors in backprop for didactic purposes

### 3.1 EXPONENTIALLY DECAYING ERROR

### 3.2 CONSTANT ERROR FLOW : NAIVE APPROACH

---
## 4 LONG SHORT-TERM MEMORY
introduce LSTM

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