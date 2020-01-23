import numpy as np

x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])
 
# print(x.shape)
# print(x.shape)

# 2. 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
 
model = Sequential()
 
# 초기 Dense 3개에서 다야몬드 꼴 모양으로 2진수의 배수로 크게 해주고 다시 줄여줌. 
# 모델의 dense층을 증가 시켜주니 출력값들이 출렁이지 않고 안정적임.
model.add(Dense(8, input_dim = 1))
model.add(Dense(32))
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(1))
 
# 3. 훈련
# metrics=['acc'] 를 mse로 변경해보자. 지표?라고 하나? 
# model.compile(loss='mse', optimizer='adam', metrics=['acc'])
# mse에서 mae로 바꿔보자. 회기모델에서는 앞 2가지 지표를 사용한다. 낮은값이 좋은지표.
# 회귀 모델은 0,000얼마 나와야 좋은 모델이다.

model.compile(loss='mse', optimizer='adam', metrics=['mse']) 
# model.fit(x, y, epochs = 100, batch_size = 20)
model.fit(x, y, epochs = 100, batch_size = 4 )  # batch_size도 2배수로 정의 

# 4. 평가 예측
# loss, mse = model.evaluate(x, y, batch_size= 1)
loss, mse = model.evaluate(x, y, batch_size = 4 )
print('mse:', mse)

# 회귀 모델 acc의 지표가 다르다. 아니 쓸수없다.  결과값이 어떻게 나오나???

x_prd = np.array([11,12,13])
# aa = model.predict(x_prd, batch_size = 1 )
aa = model.predict(x_prd, batch_size = 4 )
print('x_prd : ', aa)

# bb = model.predict(x, batch_size= 1 )
bb = model.predict(x,  batch_size = 4 )
print('x : ', bb)
# 소수점이 나와서 정확한 값이 되지 않기 떄문에 틀리다.