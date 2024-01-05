### RNN Text Classification
- Binary Classification
- Multi-Class Classification

### Text Classification using Keras(supervised learning)
- Supervised Learning(지도 학습) : Train Data로 Label이라는 정답을 포함하고 있는 Dataset을 사용하여 학습; 정답을 알고 있는 상태로 훈련
- Validation(검증) : 모든 Sample을 사용하여 학습하지 않고 일정 비율의 데이터를 남기고 이를 예측한 뒤 Label 값과 대조하여 검증하는 데 사용
- Embedding() : 각각의 단어가 정수로 변환된 값(index)를 입력으로 받아 임베딩 작업을 수행; 인덱싱 작업(정수 인코딩)의 방법 중 하나로는 빈도수에 따라 정렬
- 텍스트 분류는 RNN의 Many-to-one 문제에 속하므로, 모든 시점에 대해서 입력을 받지만, 최종시점의 은닉 상태 만을 출력
    * Binary Classification : 출력값의 종류가 두 종류일 경우(loss function = binary_crossentropy)
    * Multi-Class Classification : 출력값의 종류가 세 가지 이상(loss function : categorical_crossentropy)

```python
model.add(SimpleRNN(hidden_units, input_shape=(timesteps, input_dim)))
# hidden_units : RNN 출력의 크기; 은닉 상태의 크기
# timesteps : 시점의 수; 각 문서의 단어 수
# input_dim : 입력의 크기; 임베딩 벡터의 차원
```

---

### Tagging Task using Keras
- 지도 학습을 통해 이루어지는 분류 작업
    * Named Entity Recognition(개체명 인식) : 이름(의미)을 갖는 개체를 보고 해당 단어(개체)의 유형 파악
        + Named Entity Recognition using NTLK example
        
        ```python
        from nltk import word_tokenize, pos_tag, ne_chunk

        sentence = "James is working at Disney in London"
        
        # Tokenize the sentence and tag
        tokenized_sentence = pos_tag(word_tokenize(sentence))
        print(tokenized_sentence)

        # 개체명 인식
        ner_sentence = ne_chunk(tokenized_sentence)
        print(ner_sentence)
        ```

    * Part-of-Speech Tagging(품사 태깅) : 단어의 품사 파악
        + Part-of-speech Tagging using Bi-LSTM example
        
        ```python
        # Preprocessing
        import nltk
        import numpy as np
        import matplotlib.pyplot as plt
        from tensorflow.keras.preprocessing.text import Tokenizer
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        from tensorflow.keras.utils import to_categorical
        from sklearn.model_selection import train_test_split

        # Tokenized and Tagged Data
        tagged_sentences = nltk.corpus.treebank.tagged_sents()
        print("품사 태깅이 된 문장 개수: ", len(tagged_sentences)) # 3914

        # Diverse data into words and tags
        sentences, pos_tags = [], [] 
        for tagged_sentence in tagged_sentences:
            sentence, tag_info = zip(*tagged_sentence) # as the tagged_sentence shows tuple of word and tag of each sample, so use zip() to seperate them
            sentences.append(list(sentence))
            pos_tags.append(list(tag_info))

        def tokenize(samples):
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(samples)
        return tokenizer

        src_tokenizer = tokenize(sentences)
        tar_tokenizer = tokenize(pos_tags)

        vocab_size = len(src_tokenizer.word_index) + 1
        tag_size = len(tar_tokenizer.word_index) + 1
        print('단어 집합의 크기 : {}'.format(vocab_size))
        print('태깅 정보 집합의 크기 : {}'.format(tag_size))

        # Encoding
        X_train = src_tokenizer.texts_to_sequences(sentences)
        y_train = tar_tokenizer.texts_to_sequences(pos_tags)

        # padding into max length of encoded samples
        max_len = 150
        X_train = pad_sequences(X_train, padding='post', maxlen=max_len)
        y_train = pad_sequences(y_train, padding='post', maxlen=max_len)
        ```

        ```python
        # Generate POS Tagger
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense, LSTM, InputLayer, Bidirectional, TimeDistributed, Embedding
        from tensorflow.keras.optimizers import Adam

        embedding_dim = 128
        hidden_units = 128

        model = Sequential()
        model.add(Embedding(vocab_size, embedding_dim, mask_zero=True)) # Zero padding
        model.add(Bidirectional(LSTM(hidden_units, return_sequences=True))) # Many-to-many
        model.add(TimeDistributed(Dense(tag_size, activation=('softmax'))))

        model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(0.001), metrics=['accuracy'])
        model.fit(X_train, y_train, batch_size=128, epochs=7, validation_data=(X_test, y_test))

        # Testing 
        index_to_word = src_tokenizer.index_word
        index_to_tag = tar_tokenizer.index_word

        i = 10 # 확인하고 싶은 테스트용 샘플의 인덱스.
        y_predicted = model.predict(np.array([X_test[i]])) # 입력한 테스트용 샘플에 대해서 예측값 y를 리턴
        y_predicted = np.argmax(y_predicted, axis=-1) # 확률 벡터를 정수 레이블로 변환.

        print("{:15}|{:5}|{}".format("단어", "실제값", "예측값"))
        print(35 * "-")

        for word, tag, pred in zip(X_test[i], y_test[i], y_predicted[0]):
            if word != 0: # PAD값은 제외함.
                print("{:17}: {:7} {}".format(index_to_word[word], index_to_tag[tag].upper(), index_to_tag[pred].upper()))
        ```