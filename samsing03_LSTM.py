import numpy as np
import pandas as pd

samsung = np.load('../Study/keras/Data/samsung.npy')
kospi200 = np.load('../Study/keras/Data/kospi200.npy')

def split_xy5(dataset, time_steps, y_column):  # 426,5  // 5 //1
    x, y = list(), list()
    for i in range(len(dataset)):  #  426
        x_end_number = i + time_steps   # 0 +5 =5
        y_end_number = x_end_number + y_column  # 5+1 = 6
        
        if y_end_number > len(dataset):
            break
        tmp_x = dataset[i:x_end_number, : ]  #  0: 5
        tmp_y = dataset[x_end_number : y_end_number, 3 ]  # 5:6, 3
        x.append(tmp_x)    # 
        y.append(tmp_y)
    return np.array(x), np.array(y)

#3차원 > 2차원

x, y = split_xy5(samsung, 5, 1)
print(x.shape)
print(y.shape)
print(x[0, : ], '\n' , y[0])  # x 0번째 행, 뒤에 모든 

# 데이터 셋 나누기
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state =1, test_size = 0.3, shuffle = False
)

print(x_train.shape)
print(x_test.shape)

#데이터 전처리
# Standndard Scaler
from sklearn.preprocessing import StandardScaler

x_train = np.reshape(x_train,(x_train.shape[0], x_train.shape[1] * x_train.shape[2]))
x_test = np.reshape(x_test,(x_test.shape[0], x_test.shape[1] * x_test.shape[2]))

scaler = StandardScaler()
scaler.fit(x_train)

x_train_scaled = scaler.transform(x_train)
x_test_scaled = scaler.transform(x_test)

print(x_train_scaled[0, :])
print(x_train_scaled.shape)


# x_train_scaled = x_train_scaled.reshape(-1,5,5)
# x_test_scaled = x_test_scaled.reshape(-1,5,5)
x_train_scaled = np.reshape(x_train_scaled,(x_train_scaled.shape[0],5,5))
x_test_scaled = np.reshape(x_test_scaled,(x_test_scaled.shape[0],5,5))
print("sss  : ", _train_scaled.shape)


from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Input

input1 = Input(shape=( 5, 5 ))
dense1 = LSTM(256, activation='relu')(input1)# input_shape = (3,1)은  행은 제외된 3열을 1개씩 자르겠다라고 정의됨
dense1 = Dense(32, activation='relu')(dense1)
dense1 = Dense(64)(dense1)
dense1 = Dense(8)(dense1)
dense1 = Dense(64)(dense1)
dense1 = Dense(16, activation='relu')(dense1)
dense1 = Dense(32)(dense1)
dense1 = Dense(1)(dense1)


model.compile(loss = 'mse',
              optimizer = 'adam',
              metrics = ['mae'])

from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(patience= 20)
model.fit(x_train_scaled, y_train, validation_split=0.2, verbose=1, batch_size= 1, epochs=200, callbacks=[early_stopping])

loss, mse = model.evaluate(x_test_scaled, y_test, batch_size=1)

print( " mse :  ", mse)

y_pred = model.predict(x_test_scaled)

for i in range(5):
    print("종가 : ", y_test[i], '/ 예측가 : ', y_pred[i])