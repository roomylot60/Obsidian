## Transformer Model
- 2017년 Google에서 발표한 "Attention is all you need" 논문에서 제시한 모델
- Attention Machanism 만을 사용하여 seq2seq의 구조인 *encoder-decoder*로 구현
- RNN을 사용하지 않았음에도 성능적인 우수성
```python
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
```
