Chapter 6.  컨벌루션 신경망
========================

### 영상 인식에 특화된 심층 신경망

## Convnet
영상 인식은 분류문제에 해당된다. 이 때 영상 특징 추출을 학습 과정에 포함시킨다.
+ 컨볼루션 계층 : 입력 이미지 변환
+ 풀링 계층 : 이미지 차원 축소

### 1. 컨볼루션 계층
입력 이미지에서 고유한 특징을 부각시킨 이미지를 새로만든다.
+ Convolution Filter 사용 후 활성 함수 적용
+ 활성 함수 : ReLU , Sigmoid , tanh ...etc

이때 그냥 컨볼루션 적용시 차원이 줄어드므로 Zero padding의 기법 사용.

### 2. 풀링 계층
일종의 풀링 연산으로 이미지 차원 축소를 한다
+ Mean Pooling, Max Pooling
+ 계산량을 줄이고, 과적합 방지

[MnistConv.py](MnistConv.py)


[Prev](../ch5_Deep_Learning/deep_learning.md) / [List](../readme.md)

