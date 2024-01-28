## Interpolation
- 어떤 지점 사이의 값을 기존의 값으로부터 추정하는 것
## Spline
### Spline curve
- 주어진 복수의 제어점을 통과하는 부드러운 곡선
- 인접한 두 제어점 사이의 구간을 별도의 다항식으로 구성
### Spline interpolation
- Polynomial interpolation은 한번에 모든 데이터를 사용하여 곡선을 생성
- Piecewise polynomial interopolation은 interval에 따라 polynomial을 적용하여 original function에 근사하는 수단을 사용하나, 각 junction에서의 불연속성으로 인한 문제가 발생
- Spline interpolation은 data point를 쌍 사이의 곡선에 근사하고 모든 곡선을 합하여 최종적인 Extimated approximatino을 구성