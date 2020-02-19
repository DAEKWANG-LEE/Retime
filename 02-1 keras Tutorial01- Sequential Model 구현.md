<h2> 02-1 keras Tutorial01- Sequential Model 구현</h2>
- 선형회귀모델을 사용해서 딥러닝을 구현한다는 뜻은 선형회귀 수식 Y = WX + B를 이용하여,

  X라는 입력자료와 Y라는 출력자료가 주어졌을 때 최적의 W, B를 구한다는 뜻입니다.

  우리는 잘 정제된 X, Y 데이터를 넣어주고 노드와 레이어를 구성하여 모델을 만들어주면

  케라스 라이브러리는 알아서 최적의 W(Weight)와 B(Bias)를 찾아줍니다.

  결론은 사람은 정제된 데이터 x, y를 구하고 머신은 w, b를 구한다는 것입니다.



> 모든 딥러닝 구성은 아래의 순서로 진행된다.

1. data 구성
2. model 구성
3. model compile
4. model train
5. model evaluate & predict





<h4>1. data 구성</h4>
```python
import numpy as np
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])

print(x.shape)
print(y.shape)
```

- keras의 models과 layers라는 이름을 가진 라이브러리 내의 Sequential과 Dense라는 오브젝트를 사용하기 위해 위 모델들을 import 시켜줍니다.

- numpy의 기능들을 사용하기 위해 numpy도 import를 시켜주었고, 이제부터 numpy의 기능들을 불러올때는 간결하게 'np'만 사용하여 간단하게 불러줄 것입니다.

- 연습용이므로 입력데이터와 출력데이터는 같게 하였습니다. 

  

<h4>2. model 구성</h4>
```python
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()

model.add(Dense(5, input_dim = 1))
model.add(Dense(3))
model.add(Dense(1))

model.summary()
```

- Seqeuntial모델 오브젝트를 model이라는 변수 안에 넣고, 모델 구성을 시작합니다.

- input_dim = 1, 입력 차원이 1이라는 뜻이며 입력 노드가 한개라고 생각하면 됩니다.

  만약 x배열의 데이터가 2개라면 2, 3개라면 3으로 지정을 해줍니다.

  그 다음, 만든 시퀀스 오브젝트 model에 5개의 노드를 Dense레이어를 통해 연결해줍니다. 여기서 add를 통해 하나의 레이어를 추가해주는 것입니다.

- **Dense 레이어**는 입력과 출력을 모두 연결해주며 입력과 출력을 각각 연결해주는 가중치를 포함하고 있습니다. 입력이 3개 출력이 4개라면 가중치는 총 3X4인 12개가 존재하게 됩니다. Dense레이어는 머신러닝의 기본층으로 영상이나 서로 연속적으로 상관관계가 있는 데이터가 아니라면 Dense레이어를 통해 학습시킬 수 있는 데이터가 많다는 뜻이 됩니다.

   

- **Dense의 첫번째 인자** : 출력 뉴런(노드)의 수를 결정

- **Dense의 두번째 인자** : input_dim은 입력 뉴런(노드)의 수를 결정, 맨 처음 입력층에서만 사용

- **Dense의 세번째 인자** : activation 활성화 함수를 선택

![](C:\Users\imoon\OneDrive\바탕 화면\keras01.PNG)

만들어진 시퀀스 모델의 대략적인 모양입니다.



<h4>3. model compile</h4>
```python
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
```

- loss는 손실함수를 의미합니다. 얼마나 입력데이터가 출력데이터와 일치하는지 평가해주는 함수를 의미합니다.  여기서는 손실함수를 'mse'를 사용하겠다는 의미가 됩니다.
- mse는 평균제곱오차(mean squard error)를 의미합니다.
- 예측값과의 차이를 의마하므로 작으면 작을수록 좋은 모델이라는 의미입니다.
- optimizer는 손실 함수를 기반으로 네트워크가 어떻게 업데이트 될 지 결정합니다.
- 여기서는 adam을 사용하였습니다.



<h4>4. model train</h4>
```python
model.fit(x, y, epochs=100, batch_size=1)  
```

- 컴파일한 모델을 훈련 시킵니다. 
- epochs=100, 100번 훈련을 시킨다는 의미입니다.
- batch_size는 작업단위를 의미합니다.
- batch_size의 default값은 32입니다. 명시 해주지 않으면 머신은 32개씩 잡아서 훈련합니다.



<h4>5. model evaluate</h4>
```python
loss, mse = model.evaluate(x, y, batch_size=1)
print('loss: ', loss)
print('mse : ', mse)
```

- 배치사이즈를 1로 잡아 손실 함수를 계산합니다.



<h4>6. model predict</h4>
```python
x_prd=np.array([11, 12, 13])
y_prd=model.predict(x_prd)

print(y_prd)
```

- 예시로 [11, 12, 13]을 입력하여 예측값을 확인합니다.



[Google Colab에서 구동시켜보기](https://colab.research.google.com/github/elbicuderri/keras_tutorial/blob/master/keras_DNN_1.ipynb)



