from numpy import array
import numpy as np

def split_sequence(sequence, n_steps):
    x, y = list(), list()
    for i in range(len(sequence)):#  10개
            end_ix = i + n_steps  #  0 + 4 = 4
            if end_ix > len(sequence) -1: # 멈춰라 
                #4 > 10-1 
                break
            seq_x, seq_y = sequence[ i:end_ix], sequence[end_ix]  # 0,1,2,3 / 4,
            x.append(seq_x)
            y.append(seq_y)
    return array(x), array(y)

dataset = [10,20,30,40,50,60,70,80,90,100]
n_steps = 3
x, y = split_sequence(dataset, n_steps)

## 실습 DNN 모델 구성
for i in range(len(x)):
    print(x[i], y[i])

print(x.shape)
x = x.reshape(x.shape[0], x.shape[1],1) 
print("reshpae : ", x.shape)

# 2. 모델 구성
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size = 0.6, random_state = 66, # random_state 66번째 의 랜덤스테이트 모듈을 사용하겠다(랜덤 난수 사용).
    shuffle = False
)

x_val, x_test, y_val, y_test = train_test_split(
    x_test, y_test, test_size = 0.5, random_state = 66, # random_state 66번째 의 랜덤스테이트 모듈을 사용하겠다(랜덤 난수 사용).
    shuffle = False
) 
 
model = Sequential()

model.add(LSTM(256, activation='relu', input_shape = (3,1))) # input_shape = (3,1)은  행은 제외된 3열을 1개씩 자르겠다라고 정의됨
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(8))
model.add(Dense(16))
model.add(Dense(32))
model.add(Dense(1))

model.summary()


model.compile(loss='mse', optimizer='adam', metrics=['mse']) 
model.fit(x_train, y_train, epochs = 100, batch_size = 2,
          validation_data=(x_val, y_val) ) 

# 4. 평가 예측
loss, mse = model.evaluate(x_test, y_test, batch_size = 2 )
print('mse:', mse)

# 2열이라 추가 해준다.
x_prd = np.array([[90, 100, 110]])
x_prd = x_prd.reshape(x_prd.shape[0], x_prd.shape[1],1)
# 열로 바꿔준다. 
# x_prd = np.transpose(x_prd)

aa = model.predict(x_prd, batch_size = 2 )
print('x_prd : ', aa)