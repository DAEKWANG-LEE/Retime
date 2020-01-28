
#1. 데이터
import numpy as np
x = np.array(range(1, 101))
y = np.array(range(1, 101))

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size = 0.6, random_state = 66, # random_state 66번째 의 랜덤스테이트 모듈을 사용하겠다(랜덤 난수 사용).
    shuffle = False
)

x_val, x_test, y_val, y_test = train_test_split(
    x_test, y_test, test_size = 0.5, random_state = 66, # random_state 66번째 의 랜덤스테이트 모듈을 사용하겠다(랜덤 난수 사용).
    shuffle = False
)

# 2. 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
 
model = Sequential()
 
model.add(Dense(5, input_shape = (1,)))
model.add(Dense(2))
model.add(Dense(3))
model.add(Dense(1))

model.summary()

# 3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse']) 
model.fit(x_train, y_train, epochs = 100, batch_size = 1,
          validation_data=(x_val, y_val) ) 

# 4. 평가 예측
loss, mse = model.evaluate(x_test, y_test, batch_size = 1 )
print('mse:', mse)

x_prd = np.array([101,102,103])
aa = model.predict(x_prd, batch_size = 1 )
print('x_prd : ', aa)

bb = model.predict(x_test,  batch_size = 1 )
print('x : ', bb)



#  y_predict 만들어 주자.
y_predict = model.predict(x_test, batch_size = 1 )
#RMSE 구하기
from sklearn.metrics import mean_squared_error

def RMSE(y_test, y_predcit):
    return np.sqrt(mean_squared_error(y_test, y_predict)) # lot 를 씌어주면 RMSE가 나온다.
print("RMSE : ", RMSE(y_test, y_predict))

# R2 구하기 > 1에 근사한 값이 나와야됨.
from sklearn.metrics import r2_score

r2_y_predict = r2_score(y_test, y_predict)
print("R2 : ", r2_y_predict)