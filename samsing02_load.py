import numpy as np
import pandas as pd

samsung = np.load('../Study/keras/Data/samsung.npy')
kospi200 = np.load('../Study/keras/Data/kospi200.npy')
# print(samsung)
# print(samsung.shape)
# print(kospi200)
# print(kospi200.shape)

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


# 차원이 달라서 rsshape를 해준다.
# x_train = np.array(x_train)
# x_train = x_train.reshape(-1, 25)
# x_test = x_test.reshape(-1, 25)

x_train = np.reshape(x_train,(x_train.shape[0], x_train.shape[1] * x_train.shape[2]))
x_test = np.reshape(x_test,(x_test.shape[0], x_test.shape[1] * x_test.shape[2]))

# x_train = np.array(x_train,(x_train.shape[0],))
# x_test = np.array(x_test,(x_test.shape[0],[1]))

print(" //f ", x_train.shape)
print(" //d ", x_test.shape)

scaler = StandardScaler()
scaler.fit(x_train)
# x_train = scaler.transform(x_train)

x_train_scaled = scaler.transform(x_train)
x_test_scaled = scaler.transform(x_test)
print(x_train_scaled[0, :])
# print(" StandardScaler : ", x_train)


from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(64, activation='relu', input_shape = (25,)))
model.add(Dense(32))
model.add(Dense(128))
model.add(Dense(64))
model.add(Dense(16, activation='relu'))
model.add(Dense(8))
model.add(Dense(64, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1))

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