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
