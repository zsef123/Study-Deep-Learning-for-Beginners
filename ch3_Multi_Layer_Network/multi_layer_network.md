Chapter 3. 다층 신경망의 학습
====================

## 역전파 Backpropagation
> 출력 오차를 역순으로 이동시킨다.
+ 출력층 : δ = φ'(v)·e
> 입력층 직전까지 반복
>> 은닉층 : e = W' * δ 
>> 은닉층 : δ = φ'(v)·e
>> 은닉층 : w = w + a * δ * x 

### XOR Problem
역전파 적용 예제
[BackpropXOR.py](ch3_Multi_Layer_Network/BackpropXOR.py)

## 모멘텀
+ δw = a * δ * x
+ m = δw + βn 
+ w = w + m
+ n = m
+ w(n) = δw  + β * w(n-1) / β<1
> 이전 가중치 갱신값들이 남아 계속 반영되므로 특정 가중치 갱신 값에 영향을 적게 받는다

## 비용 함수 (Cost Function)
신경망의 지도학습이란 가중치를 조절해 학습 데이터의 오차를 줄여가는 과정
오차와 비용함수의 값이 비례한다.

+ Cross entropy E = -ln(y) if d = 1 / -ln(1 - y) if d = 0
Cross entropy가 오차에 더 민감하게 반응한다.
+ 출력층 : δ = e ( != φ'(v) ) 
+ ... 나머지는 위와 동일하다.

### 정칙화
> 비용함수에 가중치의 크기를 모두 더한 값을 새로 추가한다.
+ J = E + λ * 1/2 * ||w||^2
+ λ => 연결 가중치의 크기를 얼마나 반영할 것인가.
따라서 비용함수의 값을 낮추려면 출력 오차를 줄이고 가중치의 크기도 작아야한다.
가중치의 값이 충분히 작으면 노드 사이의 연결이 끊긴것과 같게 된다.
[BackpropCE.py](ch3_Multi_Layer_Network/BackpropCE.py)

## Cross Entropy vs XOR 
[CEvsSSE.py](ch3_Multi_Layer_Network/CEvsSSE.py)
### CE가 더 빠르다 