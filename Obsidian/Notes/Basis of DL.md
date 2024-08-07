## Notion for DL

### Neural Network; 신경망
#### 1. Perceptron
- 다수의 뉴런; 노드로부터 신호를 입력 받아 하나의 신호를 출력 
$$
\begin{flalign}
a&=w_{1}x_{1}+w_{2}x_{2}+b\\ y&=h(a)&&
\end{flalign}
$$
- 입력 신호 : $x_{1}, x_{2}$
- 가중치 : $w_{1}, w_{2}$
- 편향 : $b$
- 활성화 함수:  $h$
- 출력 신호 : $y$
#### 2. Layer
- 신경망을 구성하는 각 층
- Input layer
- Hidden layer
- Output layer
#### 3. Activation Function(활성화 함수)
- 입력 신호의 총합을 신호로 변환하는 함수
- *Sigmoid* : $h(x)=\frac{1}{1+e^{-x}}$
- *Step Function* : $h(x)=\begin{cases}0\;if\;x<0\\1\;if\;x\geq 0\end{cases}$
- *Rectified Linear Unit* : $h(x)=\begin{cases}0\;if\;x<0\\x\;if\;x\geq 0\end{cases}$
- *Identity Function* : $y=x$
- *Softmax* : $y_{k}=\frac{e^{a_{k}}}{\sum^{n}_{i=1}e^{a_{i}}}$
#### 4. Loss Fucntion
- 신경망 학습의 성능을 나타내는 지표로, 손실함수의 값을 작게하여 성능을 좋게하는 매개변수 값을 탐색
- *Mean Squared Error; MSE(평균 제곱 오차)* : $E=\frac{1}{2}\sum_{k}(y_{k}-t_{k})^{2}$
- *Cross Entropy Error(교차 엔트로피 오차)* : $E=-\sum_{k}t_{k}log\;y_{k}$
#### 5. Optimizer
### Forward Propagation(순전파)
### Backward Propagation(역전파)