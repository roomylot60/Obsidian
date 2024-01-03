**seq2seq** Model : RNN에 기반하여 **Encoder**에서 입력 시퀀스를 **context vector**라는 하나의 고정된 크기의 벡터 표현으로 압축하고, **Decoder**에서 출력 시퀀스를 생성
[문제점]
1. 하나의 고정된 크기의 벡터에 모든 정보를 압축; 정보 손실이 발생
2. RNN의 문제점 중 하나인 Vanishing gradient(기울기 소실) 발생

## Idea of Attention machanism
Decoder에서 출력 단어를 예측하는 매 time step마다, Encoder에서 전체 입력 문장을 다시 한 번 참고; 예측하려는 단어와 연관이 있는 입력 단어 부분에 좀 더 집중하여 참고

## Attention Function
- **Key-Value** 자료형; Dictionary 자료형 : Key 값을 통해 mapping 된 value 값을 찾을 수 있음
- Attention function : 주어진 Query에 대해서 모든 Key와의 유사도를 구해 Value에 반영하고, Value의 총 합(Attention value)을 return; `Attention(Q, K, V) = Attention Value`![[../Attatched/Pasted image 20240103142101.png]]
```python
Q = Query # t 시점의 decoder cell에서의 hidden state
K = Keys # 모든 시점의 encoder cell의 hidden states
V = Values # 모든 시점의 encoder cell의 hidden states
```

## Dot-Product Attention
![[../Attatched/Pasted image 20240103142716.png]]
- Attention Score : 현재 decoder의 `t 시점`에서의 단어를 예측하기 위해 encoder의 모든 hidden state의 값(`h_{i}`)이 decoder의 현재 hidden state(`s_{t}`)와 얼마나 유사한지를 판단하는 값으로, s__{t}^{T}와 h_{i}의 dot product(내적) 결과로 모두 Scalar 값; `score(s_{t}, h_{i}) =  s__{t}^{T}h_{i}` ![[../Attatched/Pasted image 20240103143640.png]](e^{t}는 모든 score의 모음값)
- Attention Distribution : e^{t}에 softmax를 적용하여 얻어낸 확률 분포(총 합은 1)
- Attention Weight : Attention Distribution의 각각의 값![[../Attatched/Pasted image 20240103144125.png]]
- Attention Value : 각 encoder의 hidden state, attention weight의 곱들을 합한 Weighted Sum; Context Vector![[../Attatched/Pasted image 20240103144422.png]]
- Concatenate Vector : Attention Function의 결과로 얻은 Attention Value; `a_{t}`와 decoder의 hidden state; `s_{t}`를 concatenate(결합)하여 얻은 벡터![[../Attatched/Pasted image 20240103145013.png]]
- 