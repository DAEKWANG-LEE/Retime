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

x1, y1 = split_xy5(samsung, 5, 1)
x2, y2 = split_xy5(kospi200, 5, 1)

print(x1.shape)
print(x2.shape)
print(y1.shape)
# print(x[0, : ], '\n' , y[0])  # x 0번째 행, 뒤에 모든 

# 데이터 셋 나누기
from sklearn.model_selection import train_test_split
x1_train, x1_test, y1_train, y1_test = train_test_split(
    x1, y1, random_state = 1, test_size = 0.3, shuffle = False
)
x2_train, x2_test, y2_train, y2_test = train_test_split( 
    x2, y2, random_state = 1, test_size = 0.3, shuffle = False
)

print(x1_train.shape)
print(x1_test.shape)
print(x2_train.shape)
# print(x2_test.shape)

# 3차원 > 2차원

x1_train = np.reshape(x1_train,(x1_train.shape[0], x1_train.shape[1] * x1_train.shape[2]))
x1_test = np.reshape(x1_test,(x1_test.shape[0], x1_test.shape[1] * x1_test.shape[2]))

x2_train = np.reshape(x2_train,(x2_train.shape[0], x2_train.shape[1] * x2_train.shape[2]))
x2_test = np.reshape(x2_test,(x2_test.shape[0], x2_test.shape[1] * x2_test.shape[2]))

print(x2_test.shape)

#데이터 전처리
# Standndard Scaler
from sklearn.preprocessing import StandardScaler
scaler1= StandardScaler()
scaler1.fit(x1_train)
x1_train_scaled = scaler1.transform(x1_train)
x1_test_scaled = scaler1.transform(x1_test)

scaler2= StandardScaler()
scaler2.fit(x2_train)
x2_train_scaled = scaler2.transform(x2_train)
x2_test_scaled = scaler2.transform(x2_test)

print(x2_train_scaled[0, :])
print(x2_train_scaled.shape)

x1_train = np.reshape(x1_train_scaled,(x1_train.shape[0], 5,5))
x1_test = np.reshape(x1_test_scaled,(x1_test.shape[0],5,5))

x2_train = np.reshape(x2_train_scaled,(x2_train.shape[0],5,5))
x2_test = np.reshape(x2_test_scaled,(x2_test.shape[0],5,5))

# x_train_scaled = x_train_scaled.reshape(-1,5,5)
# x_test_scaled = x_test_scaled.reshape(-1,5,5)
# x_train_scaled = np.reshape(x_train_scaled,(x_train_scaled.shape[0],5,5))
# x_test_scaled = np.reshape(x_test_scaled,(x_test_scaled.shape[0],5,5))


from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Input


input1 = Input(shape=(5,5))
dense1 = LSTM(256, activation='relu')(input1)# input_shape = (3,1)은  행은 제외된 3열을 1개씩 자르겠다라고 정의됨
dense1 = Dense(32, activation='relu')(dense1)
dense1 = Dense(64)(dense1)
dense1 = Dense(8)(dense1)
dense1 = Dense(64)(dense1)
dense1 = Dense(16, activation='relu')(dense1)
dense1 = Dense(32)(dense1)
output1 = Dense(25)(dense1)

input2 = Input(shape=(5,5))
dense2 = LSTM(256, activation='relu')(input2)# input_shape = (3,1)은  행은 제외된 3열을 1개씩 자르겠다라고 정의됨
dense2 = Dense(32, activation='relu')(dense2)
dense2 = Dense(64)(dense2)
dense2 = Dense(8)(dense2)
dense2 = Dense(64)(dense2)
dense2 = Dense(16, activation='relu')(dense2)
dense2 = Dense(32)(dense2)
output2 = Dense(25)(dense2)

from keras.layers.merge import concatenate

merge = concatenate([output1, output2])
output3 = Dense(1)(merge)

model = Model(inputs = [input1, input2], outputs = output3 )


model.compile(loss = 'mse',
              optimizer = 'adam',
              metrics = ['mae'])

from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(patience= 20)
model.fit([x1_train, x2_train], y1_train, 
          validation_split=0.2, 
          verbose=1, 
          batch_size= 1, epochs=50, callbacks=[early_stopping])

loss, mse = model.evaluate([x1_test,x2_test], y1_test, batch_size=1)

print( " mse :  ", mse)

y_pred = model.predict([x1_test,x2_test])

for i in range(5):
    print("종가 : ", y1_test[i], '/ 예측가 : ', y_pred[i])



from sklearn.metrics import mean_squared_error

def RMSE(y_test, y_predcit):
    return np.sqrt(mean_squared_error(y1_test, y_pred)) # lot 를 씌어주면 RMSE가 나온다.
print("RMSE : ", RMSE(y1_test, y_pred))

# R2 구하기 > 1에 근사한 값이 나와야됨.
from sklearn.metrics import r2_score

r2_y_predict = r2_score(y1_test, y_pred)
print("R2 : ", r2_y_predict)

# 돌릴때 우선 Total params:  수치를 자료의 갯수보다 많은면 불필요하다. 왜? 자료가 200개 이상인데 개산이 2번 이상이 효율이 적음.