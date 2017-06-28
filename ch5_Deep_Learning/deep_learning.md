Chapter 5. 딥러닝
===================
### 심층 신경망을 이용한 머신러닝 기법

## Backpropagation을 심층 신경망 학습시 어려움
### 1. 그래디언트 소실(Vanishing Gradient)
#### 출력층에서 멀어질수록 신경망의 오차 반영 X
+ > 노드들의 활성함수를 ReLU(Rectified Linear Unit)으로 바꾼다.
+ ReLu  = x, x > 0 / 0, x <= 0 = max(0, x)
+ ReLu' = 1, x > 0 / 0, x <= 0
[DeepReLU.py](DeepReLU.py)
### 2. 과적합
#### 은닉층이 늘어나면 모델이 복잡해져 과적합에 취약해진다.
1. Dropout 기법
    + 신경망 전체가 아닌 일부 노드만 무작위로 골라 학습시키는 기법.
    >[DeepDropout.py](DeepDropout.py)

    + 은닉층의 활성함수로 Sigmoid 사용 
2. 정칙화 항 추가
    + 과적합 해소를 위해 신경망을 최대한 단순하게 만든다.
    + 대량의 학습 데이터를 사용한다. (데이터 편향 방지)

### 3. 많은 계산량
+ GPU, 배치 정규화

[Prev](../ch4_Neural_Network_And_Classification/Neural_Network_And_Classification.md)/ [List](../readme.md) / [Next](../ch6_Convolution_NN/convolution_nn.md)
