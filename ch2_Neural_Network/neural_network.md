Chapter 2. 신경망
====================

![summery](https://t1.daumcdn.net/thumb/R1280x0/?fname=http://t1.daumcdn.net/brunch/service/user/wLl/image/XesjORemSuVCNPLihSg_4MsXduQ)
신경망 모습 요약 

+ 신경망 : 뇌의 신경세포를 모사헤 만든 노드들의 네트워크
+ 입력 -> 은닉 -> 출력 
+ 은닉층에서 활성함수로 선형함수 사용 X => 은닉층 추가 효과가 사라짐
+ 지도학습 : 신경망의 출력과 정답의 차이를 줄이도록 연결 가중치 변경 => 변경 방법 : 학습 규칙

## 델타 규칙
+ 어떤 입력 노드가 출력 노드의 오차에 기여했다면, 두 노드의 연결 가중치는 해당 입력 노드의 출력과 출력 노드의 오차에 비례해 조절한다.
![delta](https://upload.wikimedia.org/wikipedia/commons/thumb/6/60/ArtificialNeuronModel_english.png/900px-ArtificialNeuronModel_english.png)
+ w = w + a·φ'(v)·e·x       ... 일반식
+ a = 학습률 , e = 출력노드 i의 오차, v = 출력노드 i의 가중합
+ φ = 1 / (1 + e ^ -x)      ... Signoid = Activation Function
+ φ' = φ (1 - φ)
  
## SGD (Stochastic Gradient Descent)
+ 한개의 데이터마다 매번 가중치 갱신
+ 위의 일반적인 델타 규칙 적용
+ 간단한 구현 
> [DeltaSGD.py](DeltaSGD.py)

## 배치
+ 모든 학습 데이터의 오차에 관한 가중치 갱신값의 평균으로 갱신
+ 평균을 계산하는데 비용이 크다, 특정 학습 데이터에 따라 학습 편차 큼.
> [DeltaBatch.py](DeltaBatch.py)

## 미니배치
+ SGD+배치, 임의의 데이터 가중치 갱신값의 평균
> [SGD vs Batch.py](SGDvsBatch.py)

### 하지만 단층 신경망은 선형 분리 불가능 문제를 해결하지 못한다.