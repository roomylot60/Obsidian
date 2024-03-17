## 1. 모델링
### 조사한 모델
- 여러 모델에 대한 성능을 확인하기 위해 팀원 별 분류 모델 작성
- 기본 모델 : K-means 클러스터링
- 추가 모델 조사
	- 정림 :  Affinity Propagation
	- **홍기 : Gaussian Mixture Model**
	- 서연 : Mean Shift
	- 대환 : Agglomerative Clustering

```python
# GMM 적용
from sklearn.mixture import GaussianMixture
data = df.values.tolist()
# n_components로 미리 군집 개수 설정
gmm = GaussianMixture(n_components=10, n_init=3, random_state=0).fit(data)
# n_init : 모델 반복 횟수 -> 파라미터를 무작위로 선정하여 수렴할 때까지 학습
gmm_labels = gmm.predict(data)
# GMM 후 클러스터링 레이블을 따로 설정
df['gmm_cluster'] = gmm_labels
```

```powershell
2    80
9    54
7    53
4    50
1    39
3    35
8    33
6    32
0    27
5    23
Name: gmm_cluster, dtype: int64
```

- 모델 성능 지표로 실루엣 계수를 활용

```python
from sklearn.metrics import silhouette_score, silhouette_samples
score_samples = silhouette_samples(data, df['gmm_cluster'])
df['silhouette_coeff']=score_samples
average_score = silhouette_score(data, df['gmm_cluster'])
print('Silhouette Analysis Score:{0:.3f}'.format(average_score))
```

```powershell
Silhouette Analysis Score:0.133
```


## 2. 내부 함수 작성
### 추천 시스템 구조
1. 거주 선택과 관련된 요소 6가지에 대해 1차 k-means cluster를 사용하여 분류
2. 주 타켓층(MZ 세대)들이 거주지 선택에 있어서 선호하는 요소들을 사용하여 2차 분류

### 작성 코드 예시
#### 함수 구성 전
![](Attatched/tmp_1.png)
![](Attatched/tmp_2.png)

#### 함수 구성 후

- 1차 분류를 위한 features를 다루는 함수
- '교통', '교육', '육아', '치안', '건강', '편의' 6가지 요소를 활용하였으며, 각 요소에 속하는 feature에 대해 가중치를 부여하고 통합
![](Attatched/assemble_features.png)
1. 교통 : 지하철, 버스, 자전거(따릉이) 정거장 수와 관련된 가중치를 계산하여 적용 및 통합하여 하나의 feature로 재구성
2. 교육 : 초등학교, 중학교, 고등학교, 학원의 수에 대한 가중치를 부여
	- 타겟층인 MZ 세대는 신혼 부부에 속하는 연령층도 포함되기에 학군에 대한 부분이 고려됨
3. 육아 : 소아과, 어린이집에 대한 부분을 포함
4. 치안 : 경찰, 소방서 및 CCTV 설치 수 등을 고려
5. 건강 : 병원, 약국의 수
6. 편의 시설 : 백화점, 편의점, 카페, 시장 수 대해 각각의 가중치를 부여

![](Attatched/processing_df.png)
- 서울시 행정구 별 면적에 대한 데이터를 화룡
- 면적 대비 시설의 개수를 사용할 feature를 선정
- 개별 수치가 매우 큰 데이터에는 log scale을 적용
- 적용한 수치의 이상치 및 의미가 없는 값을 제거하기 위해 quantile을 적용
- 최종 사용할 데이터를 DF 형식으로 리턴

## 3. 가중치 로직 개발

- 시스템 구조 상 2차에 걸쳐 군집화를 실시하므로, 각 군집화에서 사용되는 feature들에 대해 가중치 부여를 분리
![](Attatched/Pasted%20image%2020240318034056.png)
- 1차 분류에 사용되는 feature들과  2차 분류에 사용되는 feature 간의 연관성을 고려하여 가중치를 부여
![](Attatched/Pasted%20image%2020240318033158.png)
- 2차 분류 가중치에 대한 근거로 크롤링 데이터를 활용
![](Attatched/Pasted%20image%2020240318035011.png)

