import numpy as np

#1. 데이터
x = np.array(range(1, 101))
y = np.array(range(1, 101))

x_train = x[:60]
y_train = y[:60]

x_test = x[60:80]
y_test = y[60:80]

x_val = x[80: ]
y_val = y[80: ]


# 2. 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
 
model = Sequential()
 
# 초기 Dense 3개에서 다야몬드 꼴 모양으로 2진수의 배수로 크게 해주고 다시 줄여줌. 
# 모델의 dense층을 증가 시켜주니 출력값들이 출렁이지 않고 안정적임.

# model.add(Dense(5, input_dim = 1))
model.add(Dense(5, input_shape = (1,)))
model.add(Dense(2))
model.add(Dense(3))
model.add(Dense(1))

model.summary()

# 3. 훈련
# metrics=['acc'] 를 mse로 변경해보자. 지표?라고 하나? 
# model.compile(loss='mse', optimizer='adam', metrics=['acc'])
# mse에서 mae로 바꿔보자. 회기모델에서는 앞 2가지 지표를 사용한다. 낮은값이 좋은지표.
# 회귀 모델은 0,000얼마 나와야 좋은 모델이다.

model.compile(loss='mse', optimizer='adam', metrics=['mse']) 
# model.fit(x, y, epochs = 100, batch_size = 20)
# fit은 머신이 한다. 그렇기 때문에 train에 넣는다.
model.fit(x_train, y_train, epochs = 100, batch_size = 1,
          validation_data = (x_val, y_val) )  # batch_size도 2배수로 정의    validation_data=()

# 4. 평가 예측
# loss, mse = model.evaluate(x, y, batch_size= 1)
# evaluate 은 test(사람이 값을 넣어주는것)
loss, mse = model.evaluate(x_test, y_test, batch_size = 1 )
print('mse:', mse)

# 회귀 모델 acc의 지표가 다르다. 아니 쓸수없다.  결과값이 어떻게 나오나???

x_prd = np.array([101,102,103])
# aa = model.predict(x_prd, batch_size = 1 )
aa = model.predict(x_prd, batch_size = 1 )
print('x_prd : ', aa)

# bb = model.predict(x, batch_size= 1 )
bb = model.predict(x_test,  batch_size = 1 )
print('x : ', bb)
# 소수점이 나와서 정확한 값이 되지 않기 떄문에 틀리다