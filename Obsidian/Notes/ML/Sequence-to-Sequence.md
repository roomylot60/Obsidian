## Sequence-to-Sequence; [Seq2seq](https://wikidocs.net/24996)
### Seq2seq
- 입력된 Sequence로부터 다른 도메인의 Sequence를 출력하는 모델(ex - chatBoT, 기계 번역, STT, 내용 요약)
- [RNN](ML/Recurrent_Neural_Network.md)을 조립하는 방식(하나의 RNN을 Encoder, 다른 하나를 Decoder로 구현하여 이를 연결)에 따라 구조를 생성
- Encoder와 Decoder 두 개의 모듈로 구성![](../../Attatched/Pasted%20image%2020240327203735.png)
	- Encoder : 입력 문장의 모든 단어들을 순차적으로 입력받아 마지막에 이 모든 단어 정보들을 압축한 Context vector를 생성
    * Context Vector : Encoder에서 출력하는 마지막 시점의 은닉 상태로 Decoder의 첫번째 은닉 상태에 사용
    * Decoder : 기본적 구조는 RNNLM으로 초기 입력으로 문장의 시작을 의미하는 `<sos>`가 입력되면 다음 단어를 예측하고 예측 단어를 다음 시점의 RNN 셀의 입력으로 사용하여 문장의 끝을 의미하는 심볼 `<eos>`가 예측될 때까지 반복
- 훈련 과정에서는 교사 강요를 사용하여 학습
	1. Encoder는 입력 문장의 모든 단어들을 순차적으로 입력
	2. 마지막에 이 모든 단어 정보를 압축해서 하나의 벡터를 생성(Context Vector)
	3. 이를 Decoder에서 번역된 단어를 순차적으로 출력

```python
class Seq2Seq(nn.Module): 
	def __init__(self, encoder, decoder, device): 
		super().__init__() 
		self.encoder = encoder 
		self.decoder = decoder 
		self.device = device 
		
		# encoder와 decoder의 hid_dim이 일치하지 않는 경우 에러메세지 
		assert encoder.hid_dim == decoder.hid_dim, \ 
			'Hidden dimensions of encoder decoder must be equal' 
		# encoder와 decoder의 hid_dim이 일치하지 않는 경우 에러메세지 
		assert encoder.n_layers == decoder.n_layers, \ 
			'Encoder and decoder must have equal number of layers' 
			
	def forward(self, src, trg, teacher_forcing_ratio=0.5): 
		# src: [src len, batch size] 
		# trg: [trg len, batch size] 
		
		batch_size = trg.shape[1] trg_len = trg.shape[0] # 타겟 토큰 길이 얻기 
		trg_vocab_size = self.decoder.output_dim # context vector의 차원 
		
		# decoder의 output을 저장하기 위한 tensor 
		outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device) 
		
		# initial hidden state 
		hidden, cell = self.encoder(src) 
		
		# 첫 번째 입력값 <sos> 토큰 
		input = trg[0,:] 
		
		for t in range(1,trg_len): # <eos> 제외하고 trg_len-1 만큼 반복 
			output, hidden, cell = self.decoder(input, hidden, cell) 
			
			# prediction 저장 
			outputs[t] = output 
			
			# teacher forcing을 사용할지, 말지 결정 
			teacher_force = random.random() < teacher_forcing_ratio 
			
			# 가장 높은 확률을 갖은 값 얻기 
			top1 = output.argmax(1) 
			
			# teacher forcing의 경우에 다음 lstm에 target token 입력 
			input = trg[t] if teacher_force else top1 
		
		return outputs
```

#### Encoder
![](../../Attatched/Pasted%20image%2020240327204349.png)
- LSTM 혹은 GRU로 구성되어 각 시점마다 입력 벡터와 이전 시점의 은닉 상태 값을 입력 받아 현 시점의 은닉 상태값을 출력
- 위의 과정을 순차적으로 진행하여 하나의 Sequence가 입력되었을 때 마지막에 출력 되는 은닉 상태값은 모든 시점의 영향을 받은 값이고, 해당 값을 Context vector라고 함


```python
class Encoder(nn.Module):
	def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
		super().__init__()
		self.hid_dim = hid_dim
		self.n_layers = n_layers
		# embedding: 입력값을 emb_dim 크기의 vector로 변경
		self.embedding = nn.Embedding(input_dim, emb_dim)
		# rnn: LSTM 혹은 GRU로 embedding을 받아 hid_dim 크기의 hidden state, cell state을 출력
		self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)
		self.dropout = nn.Dropout(dropout)
	def forward(self, src):
		# sre: [src_len, batch_size]
		# embedded: embedding을 통해 word2vec된 vector 값
		# 해당 vector값에 dropout을 사용하여 Overfitting(과적합)을 방지
		embedded = self.dropout(self.embedding(src))
		outputs, (hidden, cell) = self.rnn(embedded) # LSTM의 경우 hidden state, cell state 값을 출력하나, GRU의 경우 hidden state값만을 출력
		# output: [src_len, batch_size, hid_dim*n directions]
		# hidden: [n layers * n directions, batch_size, hid dim]
		# cell: [n layers * n directions, batch_size, hid dim]	
```

#### Decoder
![](../../Attatched/Pasted%20image%2020240327204753.png)
- Context vector를 최초의 은닉 상태값으로 사용하여 입력 벡터에 대한 예측값을 출력
- 예측값을 다음 시점의 은닉 상태값으로 사용
- 출력 벡터를 Softmax 함수를 통해 각 출력값에 대한 확률값을 반환

```python
class Decoder(nn.Module): 
	def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout): 
		super().__init__() 
		self.output_dim = output_dim 
		self.hid_dim = hid_dim 
		self.n_layers = n_layers 
		
		# content vector를 입력받아 emb_dim 출력 
		self.embedding = nn.Embedding(output_dim, emb_dim) 
		
		# embedding을 입력받아 hid_dim 크기의 hidden state, cell 출력 
		self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)
		
		self.fc_out = nn.Linear(hid_dim, output_dim) 
		
		self.dropout = nn.Dropout(dropout) 
	
	def forward(self, input, hidden, cell): 
		# input: [batch_size] 
		# hidden: [n layers * n directions, batch_size, hid dim] 
		# cell: [n layers * n directions, batch_size, hid dim] 
		input = input.unsqueeze(0) # input: [1, batch_size], 첫번째 input은 <SOS> 
		embedded = self.dropout(self.embedding(input)) # [1, batch_size, emd dim] 
		output, (hidden, cell) = self.rnn(embedded, (hidden, cell)) 
		# output: [seq len, batch_size, hid dim * n directions] 
		# hidden: [n layers * n directions, batch size, hid dim] 
		# cell: [n layers * n directions, batch size, hid dim] 
		
		prediction = self.fc_out(output.squeeze(0)) # [batch size, output dim] 
		return prediction, hidden, cell
```
### Bilingual Evaluation Understudy Score; BLEU Score
- 자연어 처리 태스크를 기계적으로 평가할 수 있는 방법
- 기계 번역과 사람이 직접 번역한 결과의 유사도를 통해 성능을 n-gram에 기반해 측정하여 언어에 구애받지 않고 빠른 결과를 도출
#### Unigram Precision
- Candidate(후보문)과 Reference(참조문)을 비교할 때, 후보문에서 참조문 중 한 문장에서라도 *등장한 단어의 개수의 합을 후보문의 총 Unigram(단어)의 수로 나누어 준 것*    
```python
# Tokens에서의 n-gram의 수
def simple_count(tokens, n):
	return Counter(ngrams(tokens, n))

candidate = "It is a guide to action which ensures that the military always obeys the commands of the party."
tokens = candidate.split() # Tokenize
result = simple_count(tokens, 1) # Uni-gram
print("Number of Uni-gram : ", result)
```
#### Modified Unigram Precision 
- Maximum Reference Count(하나의 Ref에서 Unigram의 최대 등장 횟수)가 기존의 단순 카운트 값보다 작을 경우에 MRC로 대체하여 얻은 정밀도
```python
def count_clip(candidate, reference_list, n):
	candidate_cnt = simple_count(candidate, n)
	M_ref_cnt_dict = dict()

	for ref in reference_list:
		ref_cnt = simple_count(ref, n) # Count n-gram in ref sentence

		for n_gram in ref_cnt:
			if n_gram in M_ref_cnt_dict:
				M_ref_cnt_dict[n_gram] = max(ref_cnt[n_gram], M_ref_cnt_dict[n_gram])
			else:
				M_ref_cnt_dict[n_gram] = ref_cnt[n_gram]
	return { n_gram: min(candidate_cnt.get(n_gram, 0), M_ref_cnt_dict.get(n_gram, 0)) for n_gram in candidate_cnt }

candidate = 'the the the the the the the'
references = [
	'the cat is on the mat',
	'there is a cat on the mat'
]
result = count_clip(candidate.split(),list(map(lambda ref: ref.split(), references)),1)
print('Modified Unigram Count : ', result)

def modified_precision(candidate, reference_list, n):
	clip_cnt = count_clip(candidate, reference_list, n) 
	total_clip_cnt = sum(clip_cnt.values()) # 분자

	cnt = simple_count(candidate, n)
	total_cnt = sum(cnt.values()) # 분모

	# 분모가 0이 되는 것을 방지
	if total_cnt == 0: 
		total_cnt = 1

	# 분자 : count_clip의 합, 분모 : 단순 count의 합 ==> 보정된 정밀도
	return (total_clip_cnt / total_cnt)

result = modified_precision(candidate.split(), list(map(lambda ref: ref.split(), references)), n=1)
print('Modified Unigram Precision : ', result)
```
#### n-gram precision 
- Unigram의 경우, 단어의 순서를 고려하지 않으므로, n-gram으로 확장
<<<<<<< HEAD
* Brevity Penalty : Candidate의 길이에 BLEU의 점수가 과한 영향을 받을 수 있는 문제로, 짧은 문장이 높은 점수를 받는 점에 주목하여 이에 panalty를 부여 
$$
Count_{clip}=min(Count, Max_Ref_Count)\\
p_{n}=\frac{\sum_{n\ gram \in Candidate}Count_{clip}(n\ gram)}{\sum_{n\ gram \in Candidate}Count(n\ gram)}\\
BP=
\begin{cases}1\;if\;c>r\\
e^{(1-r/c)}\;if\;c \leq r
\end{cases}
BLEU=BP \times exp(\sum^{N}_{n=1} w_{n}log\ p_{n})$$
=======
	* Brevity Penalty : Candidate의 길이에 BLEU의 점수가 과한 영향을 받을 수 있는 문제로, 짧은 문장이 높은 점수를 받는 점에 주목하여 이에 panalty를 부여 
$$
\begin{aligned}
&p_{n}=\frac{\sum_{n\;gram\in Candidate}Count_{clip}(n\;gram)}{\sum_{n\;gram\in Candidate}Count(n\;gram)}\\
&BP=
\begin{cases}
1\;if\;c>r\\
e^{(1-r)/c}\;if\;c\leq r
\end{cases}\\
&BLEU=BP\times exp(\sum^{N}_{n=1}w_{n}log\;p_{n})
\end{aligned}$$
>>>>>>> origin/main

```python
# Ca 길이와 가장 근접한 Ref의 길이를 리턴하는 함수
def closest_ref_length(candidate, reference_list):
	ca_len = len(candidate) # ca 길이
	ref_lens = (len(ref) for ref in reference_list) # Ref들의 길이
	# 길이 차이를 최소화하는 Ref를 찾아서 Ref의 길이를 리턴
	closest_ref_len = min(ref_lens, key=lambda ref_len: (abs(ref_len - ca_len), ref_len))
	return closest_ref_len

# closest_ref_length를 통해 r을 구해 BP를 계산
def brevity_penalty(candidate, reference_list):
	ca_len = len(candidate)
	ref_len = closest_ref_length(candidate, reference_list)

	if ca_len > ref_len:
		return 1

	# candidate가 비어있다면 BP = 0 → BLEU = 0.0
	elif ca_len == 0 :
		return 0
	else:
		return np.exp(1 - ref_len/ca_len)

def bleu_score(candidate, reference_list, weights=[0.25, 0.25, 0.25, 0.25]):
	bp = brevity_penalty(candidate, reference_list) # 브레버티 패널티, BP

	p_n = [modified_precision(candidate, reference_list, n=n) for n, _ in enumerate(weights,start=1)] 
	# p1, p2, p3, ..., pn
	score = np.sum([w_i * np.log(p_i) if p_i != 0 else 0 for w_i, p_i in zip(weights, p_n)])
	return bp * np.exp(score)

import nltk.translate.bleu_score as bleu

candidate = 'It is a guide to action which ensures that the military always obeys the commands of the party'
references = [
	'It is a guide to action that ensures that the military will forever heed Party commands',
	'It is the guiding principle which guarantees the military forces always being under the command of the Party',
	'It is the practical guide for the army always to heed the directions of the party'
]

print('실습 코드의 BLEU :',bleu_score(candidate.split(),list(map(lambda ref: ref.split(), references))))
print('패키지 NLTK의 BLEU :',bleu.sentence_bleu(list(map(lambda ref: ref.split(), references)),candidate.split()))
```