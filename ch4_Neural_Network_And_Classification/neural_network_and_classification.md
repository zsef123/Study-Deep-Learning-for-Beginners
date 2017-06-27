Chapter 4. 신경망과 분류
==========================

1. 분류 - 입력 데이터가 어느 범주에 속하는가. <= 신경망 모델에 사용
2. 회귀 - 값을 추정한다.

## 이진 분류
+ yes or no의 discrete type.
+ Sigmoid의 경우 Threshold값 적용.
+ Cost Function = Cross entropy, 활성 함수 = Sigmoid

## 다범주 분류
+ n개의 범주에서 n개의 출력노드 (n > 2)
+ 범주 n = [0...0] 에서 n번째 index 값 1 => one-hot, 1-of-N encoding
+ ex) 범주 1 = [1,0,0], 2=[0,1,0], 3=[0,0,1] => y values

### Softmax
+ 활성함수로 사용한다
+ φ = e^vi / Sigma(e^vk)
+ 모든 값의 합은 1.
+ 확률로 해석 가능하다.
+ n=2면 Sigmoid와 동일

[MultiClass](MultiClass.py)
[RealMultiClass](RealMultiClass.py) - 분류 결과 출력

## Summary
+ 이진 분류    - 출력노드 1개, 활성함수=Sigmoid, Cost=Cross Entropy
+ 다범주 분류  - 출력노드 n개, 활성함수=One-hot or 1-of-N, Cost=Cross Entropy

[Prev](../ch3_Multi_Layer_Network/multi_layer_network) / [List]( ../readme.md) / [Next](../ch5_Deep_Learning/deep_learning.md)