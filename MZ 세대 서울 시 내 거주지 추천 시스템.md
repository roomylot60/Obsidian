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

```python
def assembling_features(df):
    global tmp_df
    # 피쳐합
    tmp_df = df.copy()
    # 교통
    tmp_df['교통'] = tmp_df['SUBWAY_NUM'] + 0.93 * tmp_df['BUS_CNT'] + 0.06 * tmp_df['BIKE_NUM']
    tmp_df = tmp_df.drop(['SUBWAY_NUM', 'BUS_CNT', 'BIKE_NUM'], axis=1)

    # 교육
    tmp_df['교육'] = (0.07) * tmp_df['MID_SCH_NUM'] + (0.03) * tmp_df['HIGH_SCH_NUM'] + tmp_df['ACADEMY_NUM'] * (0.7) + (
        0.9) * tmp_df['ELE_SCH_NUM']
    tmp_df = tmp_df.drop(['MID_SCH_NUM', 'HIGH_SCH_NUM', 'ACADEMY_NUM', 'ELE_SCH_NUM'], axis=1)

    # 육아
    tmp_df['육아'] = tmp_df['CHILD_MED_NUM'] + tmp_df['KINDER_NUM']
    tmp_df = tmp_df.drop(['CHILD_MED_NUM', 'KINDER_NUM'], axis=1)

    # 치안
    tmp_df['치안'] = tmp_df['POLICE_NUM'] + tmp_df['CCTV_NUM'] + tmp_df['FIRE_NUM']
    tmp_df = tmp_df.drop(['POLICE_NUM', 'CCTV_NUM', 'FIRE_NUM'], axis=1)

    # 건강
    tmp_df['건강'] = (0.94) * tmp_df['HOSPITAL_NUM'] + tmp_df['PHARM_NUM']
    tmp_df = tmp_df.drop(['HOSPITAL_NUM', 'PHARM_NUM'], axis=1)

    # 편의시설
    tmp_df['편의시설'] = 0.04 * tmp_df['DPTM_NUM'] + 0.44 * tmp_df['CON_NUM'] + 0.25 * tmp_df['CAFE_NUM'] + 0.27 * tmp_df[
        'RETAIL_NUM']
    tmp_df = tmp_df.drop(['DPTM_NUM', 'CON_NUM', 'CAFE_NUM', 'RETAIL_NUM'], axis=1)

    tmp_df.set_index('DONG_CODE', inplace=True)

    return tmp_df

```

1. 교통 : 지하철, 버스, 자전거(따릉이) 정거장 수와 관련된 가중치를 계산하여 적용 및 통합하여 하나의 feature로 재구성
2. 교육 : 초등학교, 중학교, 고등학교, 학원의 수에 대한 가중치를 부여
	- 타겟층인 MZ 세대는 신혼 부부에 속하는 연령층도 포함되기에 학군에 대한 부분이 고려됨
3. 육아 : 소아과, 어린이집에 대한 부분을 포함
4. 치안 : 경찰, 소방서 및 CCTV 설치 수 등을 고려
5. 건강 : 병원, 약국의 수
6. 편의 시설 : 백화점, 편의점, 카페, 시장 수 대해 각각의 가중치를 부여

```python
def preprocessing_df():
    area_df = merge_area_data()
    assem_df = assembling_features(area_df)

    tmp_data = assem_df.iloc[:, 3:]
    df = tmp_data.div(assem_df['AREA'], axis=0)

    max_lim_log_list = ["교통", "치안", "교육", "COLIVING_NUM", "STARBUCKS_NUM", "MC_NUM", "NOISE_VIBRATION_NUM", "VEGAN_CNT"]

    for f in max_lim_log_list:
        quan = df[f].quantile(0.95)
        df[f] = np.where(df[f] > quan, quan, df[f])
        df[f] = np.log1p(df[f])

    max_lim_list = ["LEISURE_NUM", "GOLF_NUM", "건강", "편의시설"]
    for f in max_lim_list:
        quan = df[f].quantile(0.95)
        df[f] = np.where(df[f] > quan, quan, df[f])

    ro_df = robust_scaling(df)
    ro_df = ro_df[['교통', '치안', '건강', '편의시설', '교육',
             '육아', 'MZ_POP_CNT', 'COLIVING_NUM', 'VEGAN_CNT', 'KIDS_NUM',
             'PARK_NUM', 'STARBUCKS_NUM', 'MC_NUM', 'NOISE_VIBRATION_NUM',
             'SAFE_DLVR_NUM', 'LEISURE_NUM', 'GYM_NUM', 'GOLF_NUM', 'CAR_SHR_NUM',
             'ANI_HSPT_NUM']]

    return ro_df
```

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

#### 가중치 함수 : 사용자 입력값과 내부 가중치 결합

```python
def weighting(user_df, df, select, user_name):
    weight_df = pd.read_excel('recommend_app/data/1107_가중치.xlsx')
    weight_df.rename(columns={'Unnamed: 0': '분류'}, inplace=True)
    weight_df.fillna(0, inplace=True)
    weight_df.set_index('분류', inplace=True)

    values = user_df.loc[user_name].values
    weight = weight_df[weight_df.columns].values
    w = [1] * len(weight)
    for i in range(len(weight)):
        if(select[i] == 1):
            for k in range(len(weight[i])):
                w[i] += weight[i][k]

    weighted_user_data = []
    for i in range(len(values)):
        weighted_data = values[i] * w[i]
        weighted_user_data.append(weighted_data)
    weighted_user_df = pd.DataFrame(weighted_user_data,index=df.columns,columns=['user']).T
    return weighted_user_df
```

- `.xlsx` 파일로 작성한 가중치를 데이터 프레임 형태로 입력 받아 재구성
- `user_df` : 사용자가 입력한 수치
- `weight_df` : 내부 가중치
- `weighted_user_df` : 추천 시스템에서 사용할 최종 가중치

