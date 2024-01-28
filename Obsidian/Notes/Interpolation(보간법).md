## Interpolation
![](../Attatched/Pasted%20image%2020240128171548.png)
- 어떤 지점 사이의 값을 기존의 값으로부터 추정하는 것
### Polynomial interpolation
- 한번에 모든 데이터를 사용하여 곡선을 생성
### Piecewise polynomial interopolation
- Interval에 따라 polynomial을 적용하여 original function에 근사하는 수단을 사용
- 각 junction에서의 불 연속성, 미분 불가능
## Spline
### Spline curve
- 주어진 복수의 제어점을 통과하는 부드러운 곡선
- 인접한 두 제어점 사이의 구간을 별도의 다항식으로 구성
#### Leanier spline interpolation(1차)
![](../Attatched/Pasted%20image%2020240128171525.png)
- Data point에 대해 인접한 두 점 사이를 무조건 직선으로 표현하여 추정
- 두 점 사이의 기울기를 바탕으로 선형식을 구성
#### Cubic spline interpolation(3차)
![](../Attatched/Pasted%20image%2020240128171519.png)
- 3차 다항식을 사용하여 점 사이의 데이터를 근사
- 인접한 두 data point 양 쪽에 data를 하나씩 더 추가하여 4개의 point를 3차 다항식으로 표현하여 smooth한 식을 구성
- 
### Spline interpolation
- Data point를 쌍 사이의 곡선에 근사하고 모든 곡선을 합하여 최종적인 Extimated approximation을 구성
### Ref.
https://hofe-rnd.tistory.com/entry/interpolation-Spline-method-1
