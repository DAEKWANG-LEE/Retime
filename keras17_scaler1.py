#정제된 data 를 만들어야됨. 가보자.
from numpy import array
from keras.models import Sequential
from keras.layers import Dense, LSTM

x = array([
           [1,2,3,],[2,3,4],[3,4,5],[4,5,6],[5,6,7],
           [6,7,8],[7,8,9],[8,9,10],[9,10,11],[10,11,12],
           [20000, 30000, 40000],[30000, 40000, 50000],
           [40000, 50000, 60000],[100, 200, 300]])
y = array(
            [4,5,6,7,8,9,10,11,12,13,50000,60000,70000,400])


from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import RobustScaler, MaxAbsScaler

# # 훈련 시켜라
# scaler = MinMaxScaler()
# scaler.fit(x)
# # 훈련시킨걸 변환하라
# x = scaler.transform(x)
# print(x)

# scaler = StandardScaler()
# scaler.fit(x)
# # 훈련시킨걸 변환하라
# x = scaler.transform(x)
# print(x)


scaler = StandardScaler()
scaler.fit(x)
x = scaler.transform(x)
scaler = MinMaxScaler()
scaler.fit(x)
x = scaler.transform(x)
# print(x)

# train은 10개, 나머지는 test
# Dense 모델 구현
# (14,3)
# R2 지표를 잡고.
# [250, 260, 270] 으로 predict 할것.

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size = 0.1, random_state = 66,
    shuffle = False
)
x_val, x_test, y_val, y_test = train_test_split(
    x_test, y_test, test_size = 0.9, random_state = 66,
    shuffle = False
)

model = Sequential()

model.add(Dense(16, input_shape = (3,)))
model.add(Dense(32))
model.add(Dense(8))
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(32))
model.add(Dense(32))
model.add(Dense(1))

# model.summary()

model.compile(loss = 'mse',
              optimizer = 'adam',
              metrics = ['mae'])
model.fit(x_train, y_train, 
          epochs = 100, 
          batch_size = 2,
          validation_data = (x_val, y_val))

loss, mae = model.evaluate(x_test, y_test, 
                           batch_size = 2)

import numpy as np
x_prd = np.array([[201,202,203]])

scaler = StandardScaler()
scaler.fit(x_prd)
x_prd = scaler.transform(x)
scaler = MinMaxScaler()
scaler.fit(x_prd)
x_prd = scaler.transform(x_prd)


# x_prd = np.transpose(x_prd)

aa = model.predict(x_prd, batch_size = 1 )
print('predict : ', aa)

#  y_predict 만들어 주자.
y_predict = model.predict(x_test, batch_size = 1 )
print('y_predict : ', y_predict)

# R2 구하기 > 1에 근사한 값이 나와야됨.
from sklearn.metrics import r2_score

r2_y_predict = r2_score(y_test, y_predict)
print("R2 : ", r2_y_predict)