## Deep Learning
- Artificial Neural Network(인공 신경망)의 층을 연속적으로 쌓아올려 데이터를 학습
- 기계가 가중치를 스스로 찾아내도록 자동화 시키는 심층 신경망의 학습

## Conception
### Perceptron(퍼셉트론)
- 초기 형태의 인공 신경망
- 입력값 x 벡터에 대해 가중치 w를 곱하고 편향 b(-임계값)을 더해 결과값 y를 도출

### Single-Layer Perceptron
- Input layer(입력층)과 Output layer(출력층)으로 구성

### Multi-Layer Perceptron
- Input, Output layer 사이에 Hidden layer(은닉층)을 두어 좀 더 복잡한 문제에 대응
    * Feed-Forward Neural Network(FFNN) : 입력층에서 출력층방향으로 연산이 전개
    * RNN : 은닉층의 출력값을 출력층과 은닉층의 입력으로 사용
    * Fully-connected Layer : 어떤 층의 모든 뉴런이 이전 층의 모든 뉴런과 연결된 층
    * Activation Function(활성화 함수) : 은닉층과 출력층의 뉴런에서 출력값을 결정하는 비선형 함수
        + Step Function(계단 함수)
        + Sigmoid Function : 시그모이드 함수를 활성화 함수로 하는 인공 신경망을 다층으로 쌓을 때 역전파 과정에서 사용하는 경사 하강법에 의해 Vanishing Gradient(기울기 소실)이 발생하여 학습이 잘 이루어지지 않음
        + Hyperbolic tangent function
        + ReLU Function : 음수 입력에 대해 0을 출력하고, 양수는 입력값을 출력
        + Leaky ReLU
        + Softmax Function : 시그모이드 함수처럼 출력층에서 주로 사용되며, 시그모이드가 이진 분류에 사용되는 반면, 소프트맥스는 다중 클래스 분류에 주로 사용

### Forward Propagation(순전파)
- 입력층에서 출력층 방향으로 연산을 진행하는 과정

### BackPropagation(역전파)
- 순전파 과정을 진행하여 예측값과 실제값의 오차를 계산하였을 때, 경사 하강법을 사용하여 학습률을 반영해 가중치를 업데이트하는 과정

### Concept
- Loss function : 실제값과 예측값의 차이를 수치화 해주는 함수
    * MSE
    * Binary Cross-Entropy
    * Categorical Cross-Entropy
- Batch : 가중치 등의 매개 변수의 값을 조정하기 위해 사용하는 데이터의 양
    * Batch Gradient Descent : Optimizer 중 하나로 오차를 구할 때 전체 데이터를 고려
    * Stochastic GD : 랜덤으로 선택한 하나의 데이터에 대해서만 경사 하강법을 계산
    * Mini-Batch GD : 적당한 수의 배치 크기를 지정하여 GD를 실행
- Optimizer : Compile 과정에서 사용
    * Momentum : GD에서 계산된 접선의 기울기에 한 시점 전의 접선의 기울기 값을 일정한 비율만큼 반영
    * Adagrad : 각각의 매개변수에 서로 다른 학습률을 적용
    * RMSprop : Adagrad의 학습률이 지나치게 떨어지는 단점을 보완하기 위해 사용
    * Adam : RMSprop과 Momentum을 합친 방식
- Epoch : 인공 신경망에서 전체 데이터에 대해서 순전파와 역전파가 끝난 상태로 훈련 및 검증 과정이 한차례 끝난 것
- Gradient Vanishing : 역전파 과정에서 입력층으로 갈 수록 기울기가 점차적으로 작아져 입력층에 가까운 층들에서의 업데이트가 제대로 이루어지지 않는 상태
- Gradient Exploding : 기울기가 점차 커져서 가중치들의 값이 발산
    * Gradient Clipping : 기울기 폭주를 막기 위해 임계값을 넘지 않도록 값을 제한
    * Weight initialization
        + Xavier Initialization : 이전 층과 다음 층의 뉴런의 개수를 고려하여, 여러 층의 기울기 분산 사이에 균형을 맞춰서 특정 층이 너무 주목을 받거나 다른 층이 뒤쳐지는 것을 방지(S자 형태의 activation function에서 좋은 성능) ![Xavier initialization_uniform](./img/Xavier_uniform.jpg) ![Xavier initialization_normal](./img/Xavier_normal.jpg)
        + He Initialization : 이전 층의 뉴런의 개수를 반영하여 초기화(ReLU 계열 함수에 효율적) ![He initialization_uniform](./img/He_uniform.jpg) ![He initialization_normal](./img/He_normal.jpg)
    * Batch Normalization(배치 정규화) : 인공 신경망에 들어가는 각 입력을 평균과 분산으로 정규화
        + Internal Covariate Shift(내부 공변량 변화) : 학습에 의해 가중치가 변화하면, 입력 시점에 따라 입력 데이터의 분포가 변화하는 것
    * Layer Normalization(층 정규화)

### 과적합을 방지하는 방법
1. 데이터의 양을 늘리기 : 데이터의 양이 적을 때, 데이터의 특정 패턴이나 노이즈까지 쉽게 암기할 수 있으므로, 데이터의 양을 늘려 일반적인 패턴을 학습(ex - Data Augmentation, Back Translation)

2. 모델의 복잡도 줄이기 : 은닉층의 수, 매개변수의 수를 줄여서 사용

3. Regularization(가중치 규제) 적용하기
    - L1 regularization : 가중치 w들의 절댓값의 합을 비용함수에 추가
    - L2 regularization : 모든 가중치 w들의 제곱합을 비용함수에 추가
4. Dropout : 학습 과정에서 설정한 비율로 랜덤한 신경망을 사용하지 않는 방법으로, 특정 뉴런 또는 특정 조합에 너무 의존적으로 되는 것을 방지

---

## Keras
- Preprocessing
    * Tokenizer() : 토큰화, 정수 인코딩을 통해 단어 집합 생성
    
    ```python
    from tensorflow.keras.preprocessing.text import Tokenizer

    tokenizer = Tokenizer()
    train_text = "The earth is an awesome place live"

    # Vocabulary dictionary
    tokenizer.fit_on_texts([train_text])

    # Index encoding
    sub_text = "The earth is an great place live"
    sequences = tokenizer.texts_to_sequences([sub_text])[0]

    print("Encoding : ", sequences)
    print("Vocabulary Set : ", tokenizer.word_index)
    ```

    ```bash
    정수 인코딩 :  [1, 2, 3, 4, 6, 7]
    단어 집합 :  {'the': 1, 'earth': 2, 'is': 3, 'an': 4, 'awesome': 5, 'place': 6, 'live': 7}
    ```
    
    * texts_to_matrix() : 입력된 텍스트 데이터를 행렬로 변환
        + binary : 해당 단어의 존재 여부
        + count : DTM 생성
        + freq : 각 문서에서 각 단어의 등장 횟수 / 각 문서의 모든 단어의 개수
        + tfidf : TF-IDF 행렬 생성
    * pad_sequences() : 샘플의 길이가 서로 다를 때, padding을 거쳐 필요한 부분을 0으로, 필요없는 부분은 제거하여 동일하게 맞춤
    
    ```python
    from tensorflow.keras.preprocessing.sequence import pad_sequence

    print(pad_sequences([[1,2,3],[3,4,5,6],[7,8]], maxlen=3, padding='pre'))
    ```

    ```bash
    [[1 2 3]
    [4 5 6]
    [0 7 8]]
    ```

- Word Embedding : 텍스트 내의 단어들을 dense vector로 변환하는 것으로, 학습을 통해 값이 변화 ![One-hot VS Embedding](./img/one_hot_vs_w_embedding.jpg)
    * Embedding(vocab_size, embedding_dim, input_length) : 단어를 밀집 벡터로 변환(embedding layer 생성)

- Modeling
    * Sequential() : 모델에서 사용할 layer 생성을 위한 선언
    * Dense(output_dim, input_dim, activation_function) : layer 추가 시, 해당 layer의 정보
    
    ```python
    from tensorflow.keras.models import Sequential

    model = Sequential()
    model.add(Embedding(8, 4, input_length=3)) # Adding embedding layer
    model.add(Dense(5, input_dim=4, activation='relu')) # Adding fully-connected layer
    model.add(Dense(1, activation='sigmmoid')) # Adding output layer pass 2nd parameter as you declared pre-layer output_dim,  5.
    ```
    
    * summary() : 모델의 정보를 요약

- Compile & Training
    * compile(optimizer, loss, metrics) : 모델을 기계가 이해할 수 있도록 loss_functino, optimizer, matric_function(평가 지표 함수)의 정보와 함께 compile
    * fit(train_data, label_data, epochs, batch_size, (validation_data,) verbose, validation_split) : 모델의 학습을 진행
        + validation_data(x_val, y_val) : 검증 데이터를 활용
        + validation_split : 훈련 데이터의 일정 비율을 검증 데이터로 활용
        + verbose : 1일 때, 훈련의 진행도를 나타내는 진행 막대 표시, 2일 때, 미니 배치마다 손실 정보를 출력
    
    ```python
    from tensorflow.keras.layers import SimpleRNN, Embedding, Dense
    from tensorflow.keras.models import Sequential

    vocab_size = 10000
    embedding_dim = 32
    hidden_units= 32

    # Declare Model
    model = Sequential()
    model.add(Embedding(vocab_size, embedding_dim))
    model.add(SimpleRNN(hidden_units))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
    
    # Training
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0, validation_split=0.2)
    
    # Evaluation
    model.evaluate(X_test, y_test, batch_size=32)

    # Prediction
    model.predict(X_input, batch_size=32)

    # Save
    model.save("model_name.h5")

    # Load
    from tensorflow.keras.models import load_model
    model2 = load_model("model_name.h5")
    ```

- Evaluation & Prediction
    * evaluate(test_data, test_label, batch_size)
    * predict(data, batch_size)
- Save & Load
    * save("model_path + model_name") : hdf5 file로 저장
    * load_model("model_path + model_name") : 저장 된 모델 로드

### Functional API
Layer 생성에 있어서 각 층을 일종의 함수로서 정의하고 연산자를 통해 신경망을 설계
- FFNN

```python
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

inputs = Input(shape=(10,)) # 10개의 입력을 받는 입력층
hidden1 = Dense(64, activation='relu')(inputs)
hidden2 = Dense(64, activation='relu')(hidden1)
output = Dense(1, activation='sigmoid')(hidden2) # input->hidden1->hidden2->ouptut 으로 구성

# 모델 생성 및 학습 진행
model = Model(inputs=inputs, ouputs=output)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])
model.fit(data, labels)
```

- Linear Regression

```python
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras import optimizers
from tensorflow.keras.models import Model

X = [1, 2, 3, 4, 5, 6, 7, 8, 9] # 공부하는 시간
y = [11, 22, 33, 44, 53, 66, 77, 87, 95] # 각 공부하는 시간에 맵핑되는 성적

inputs = Input(shape=(1,))
output = Dense(1, activation='linear')(inputs)
linear_model = Model(inputs, output)

sgd = optimizers.SGD(lr=0.01)

linear_model.compile(optimizer=sgd, loss='mse', metrics=['mse'])
linear_model.fit(X, y, epochs=300)
```

- Logistic Regression

```python
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

inputs = Input(shape=(3,))
output = Dense(1, activation='sigmoid')(inputs)
logistic_model = Model(inputs, output)
```
- RNN;Recurrence Neural Network -> 은닉층 생성

```python
from tensorflow.keras.layers import Input, Dense, LSTM
from tensorflow.keras.models import Model

inputs = Input(shape=(50,1))
lstm_layer = LSTM(10)(inputs)
x = Dense(10, activation='relu')(lstm_layer)
output = Dense(1, activation='sigmoid')(x)

model = Model(inputs=inputs, outputs=output)
```

### Sub-Classing API
- Linear Regression

```python
import tensorflow as tf

# 선형 회귀에 대한 class 정의
class LinearRegression(tf.keras.Model):
    # 모델의 구조 및 동작을 정의하는 생성자 정의
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear_layer = tf.keras.layers.Dense(1, input_dim=1, activation='linear')

    # 데이터를 입력 받아 예측값을 리턴하는 forward 연산
    def call(self, x):
        y_pred = self.linear_layer(x)
        return y_pred

# 모델 생성
model = LinearRegression()

# 데이터
X = [1, 2, 3, 4, 5, 6, 7, 8, 9] # 공부하는 시간
y = [11, 22, 33, 44, 53, 66, 77, 87, 95] # 각 공부하는 시간에 맵핑되는 성적

sgd = tf.keras.optimizers.SGD(lr=0.01)
model.compile(optimizer=sgd, loss='mse', metrics=['mse'])
model.fit(X, y, epochs=300)
```

---

## Neural Network Language Model; NNLM(신경망 언어 모델)
- NNLM : 자연어 학습에 대해 과거의 통계적인 접근방식(SLM)이 아닌, 인공 신경망을 사용하는 방식으로(ex - FFNNLM, RNNLM, BiLM)
    * 앞의 모든 단어를 참고하는 것이 아니라 정해진 개수의 단어(window)만을 참고하여 해당 window의 one-hot vector를 생성
    * 입력된 one-hot vector들을 은닉층 중 활성화 함수가 존재하지 않는 투사층(projection layer)에서 가중치 행렬과 곱하여 lookup table을 생성
    * Lookup table의 값은 최초에는 랜덤한 값을 가지지만 학습을 거치면서 값이 변경
    * 테이블의 각각의 값, Embedding vector는 은닉층을 거쳐 활성화 함수를 통해 window의 크기;`V`와 같은 차원의 벡터를 출력
    * 출력층에서 활성화 함수(softmax)를 거쳐 0~1 사이의 값을 갖고 총 합이 1인 `V`차원의 벡터를 손실함수로 cross-entropy를 사용해 예측값을 결정

### N-gram LM
- Language Model : 문장에 확률을 할당하는 모델
- Language Modeling : 주어진 문맥으로부터 모르는 단어를 예측하는 것
- N-gram LM : 바로 앞의 `n-1`개의 단어를 참고하여 `n`번째 단어를 예측하는 모델

## Recurrent Neural Network; RNN
- RNN : 입력과 출력을 sequence(묶음) 단위로 처리하는 모델

### RNN(순환 신경망)
- FFNN : 은닉층에서 활성화 함수를 거친 값이 출력층 방향으로만 진행
- RNN : 은닉층에서 활성화 함수(`tanh`를 주로 사용)를 거친 값이 출력층 혹은 은닉층 노드로 진행
    * Memory Cell;RNN Cell : 은닉층에서 activation function을 거친 값을 내보내면서 이전 시점의 값을 기억하고 이를 입력으로 사용하는 노드
    * Hidden State : t시점의 메모리 셀이 t+1 시점으로 보내는 값
    * RNN Architecture : 각 시점에서 입력과 출력이 동시 혹은 하나만 이루어 지는지에 따라 모델의 종류(구조)가 변함 ![RNN Models](./img/rnn_models.jpg)
        + One-to-Many : Image Captioning
        + Many-to-One : Sentiment Classification, Spam Detection
        + Many-to-Many : Chatbot, Translator, Tagging task

### RNN with Keras

```python
from tensorflow.keras.layers import SimpleRNN

# 추가 인자를 활용한 RNN 층 생성
model.add(SimpleRNN(hidden_units, input_shape=(timesteps, input_dim)))
# model.add(SimpleRNN(hidden_units, input_length=M, input_dim=N)) # 다른 표기
```

- hidden_units : 은닉 상태의 크기를 정의; 메모리 셀이 다음 시점의 메모리 셀과 출력층으로 보내는 값의 크기(ouptut_dim)
- Input : 3D tensor로 입력
    * batch_size : 한번에 학습하는 데이터의 수; sample의 수
    * input_dim : 입력 벡터의 차원 수;입력 단어 하나의 길이
    * timesteps : 입력 시퀀스의 길이(input_length);각 시점에서 단어 하나를 입력으로 받기에, 단어의 모음인 시퀀스의 길이라고 할 수 있음
- Output : return_sequences의 값에 따라 2D, 3D 결정
    * batch_size
    * output_dim : 최종 시점의 은닉 상태인 출력 벡터의 차원 수;출력 단어 하나의 길이
    * timesteps : 메모리 셀의 각 시점의 은닉 상태값(hidden states)
- RNN example

```python
import numpy as np

timestpes = 10 # 시퀀스의 길이
input_dim = 4 # 단어 벡터의 차원
hidden_units = 8 # 은닉 상태의 크기 = 메모리 셀의 용량

inputs = np.random.random((timestpes, input_dim))
hidden_state_t = np.zeros((hidden_units,)) # 초기 은닉 상태 초기화
print("Initial Hidden States :", hidden_state_t)

Wx = np.random.random((hidden_units, input_dim)) # 입력에 대한 가중치 2D 텐서 생성
Wh = np.random.random((hidden_units, hidden_units)) # 은닉 상태에 대한 가중치 2D 텐서 생성
b = np.random.random((hidden_units,)) # bias 생성

print("Wx, Wh, b : {}\n\n{}\n\n{}\n\n".format(Wx, Wh, b))
print("Wx, Wh, b shapes : ",np.shape(Wx), np.shape(Wh), np.shape(b))
print("\n")

total_hidden_states = [] # 모든 시점의 은닉 상태

# dot(arr1, arr2) : 벡터 계산 (arr1, arr2의 원소값들의 곱의 합)
for input_t in inputs:
    output_t = np.tanh(np.dot(Wx, input_t) + np.dot(Wh, hidden_state_t))
    total_hidden_states.append(list(output_t))

total_hidden_states = np.stack(total_hidden_states, axis=0)
print(total_hidden_states)
```

- output

```bash
Initial : [0. 0. 0. 0. 0. 0. 0. 0.]
Wx, Wh, b : [[0.85143664 0.97714713 0.65502162 0.28371238]
 [0.30802615 0.09059362 0.1076857  0.25372438]
 [0.28704896 0.60082044 0.75228122 0.68312046]
 [0.68097992 0.80925337 0.0645649  0.12324691]
 [0.38990914 0.92962164 0.10098862 0.67962663]
 [0.94462422 0.01337485 0.03662331 0.03050986]
 [0.28653855 0.32739224 0.49828274 0.82087065]
 [0.96042989 0.83653596 0.07487261 0.09113481]]

[[0.68096678 0.58803731 0.57988288 0.92890096 0.09135938 0.99057804
  0.01872872 0.4546827 ]
 [0.13087967 0.76399839 0.56236402 0.69592914 0.36460709 0.72990083
  0.45858138 0.00552776]
 [0.55843321 0.64096323 0.53645843 0.79040003 0.42508579 0.30223451
  0.5627866  0.86003013]
 [0.4413374  0.97321599 0.84578887 0.25420944 0.73637994 0.391908
  0.17252162 0.76213319]
 [0.06361856 0.73890545 0.58717524 0.45134905 0.05404353 0.4730816
  0.55602838 0.81121357]
 [0.56893496 0.79008566 0.87842282 0.00705078 0.06712543 0.27438764
  0.40791329 0.50191709]
 [0.18485617 0.67680875 0.32107997 0.67353141 0.58586511 0.80557961
  0.9308333  0.33228672]
 [0.58482391 0.67038387 0.61891013 0.40433411 0.7752712  0.49077138
  0.22612616 0.23695048]]

[0.77085078 0.46451979 0.05440115 0.82049665 0.09553835 0.78358678
 0.47607443 0.87411462]


Wx, Wh, b shapes :  (8, 4) (8, 8) (8,)


[[0.9311158  0.28819712 0.81778765 0.79569775 0.80725949 0.42711984
  0.64165978 0.84558929]
 [0.93584739 0.33438469 0.79853643 0.8403642  0.84959694 0.54169853
  0.64857779 0.88908515]
 [0.89903488 0.45231785 0.85710553 0.67561648 0.77069602 0.63738225
  0.81731435 0.77063852]
 [0.75156127 0.33641803 0.63492503 0.51016977 0.51385085 0.59919868
  0.59905286 0.63723528]
 [0.88905473 0.44012962 0.8013277  0.70813697 0.75661072 0.68661386
  0.76037359 0.80598781]
 [0.95834118 0.44270473 0.89593064 0.82501953 0.8508026  0.65423909
  0.80873311 0.88625939]
 [0.93018297 0.50930848 0.90068338 0.78150259 0.8930009  0.63937644
  0.8779043  0.84691315]
 [0.82701004 0.24378468 0.78295419 0.45492988 0.48932459 0.31174412
  0.63485924 0.52788644]
 [0.9632202  0.51884557 0.91992141 0.85340389 0.90910772 0.70188364
  0.87507954 0.90729884]
 [0.89504751 0.38634984 0.83208148 0.62738128 0.65260641 0.60001641
  0.74679467 0.73196294]]
```

- Deep RNN : 은닉층의 개수(깊이)가 복수인 경우

```python
model = Sequential()
# 첫번째 은닉층을 정의할 때, 각 시점의 값을 다음 은닉층으로 값을 전달하기 위해 retrun_sequences 값을 True로 설정
model.add(SimpleRNN(hidden_units, input_length=10, input_dim=5, return_sequences=True))
model.add(SimpleRNN(hidden_units, return_sequences=True))
```

- Bidirectional RNN(양방향 RNN) : 시점 t에서의 출력값을 예측할 때, 이전 시점의 입력뿐만 아니라, 이후 시점의 입력 또한 에측에 기여할 수 있다는 아이디어를 기반으로 기본적으로는 두 개의 메모리 셀을 사용하여 첫번째에서는 Forward States를, 두번째에서는 Backward States를 전달 받아 은닉상태를 계산