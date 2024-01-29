## Selective Search Algorithm
- 객체와 주변 간의 색감(color), 질감(texture) 차이, 다른 물체에 둘러 쌓여 있는 지 여부(enclosed)를 파악해서 인접한 유사한 pixel끼리 묶어 물체의 위치를 파악
![](../Attatched/Pasted%20image%2020240129155018.png)

- Bounding box들을 작고 Random하게 많이 생성
- Hierarchical grouping Algorithm을 사용해 조금씩 Merging
- Regions Of Interest; ROI 라는 영역을 제안