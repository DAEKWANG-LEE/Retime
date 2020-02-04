from numpy import array
import numpy as np

def split_sequence3(sequence, n_steps):
    x, y = list(), list()
    for i in range(len(sequence)):      #  10개
            end_ix = i + n_steps        #  0 + 4 = 4
            if end_ix > len(sequence) -1:  # -1: # 멈춰라 
                #4 > 10-1 
                break
            seq_x, seq_y = sequence[ i:end_ix, : ], sequence[end_ix , ]  # 0,1,2,3 / 4,
            x.append(seq_x)
            y.append(seq_y)
    return array(x), array(y)

in_seq1 = array([10,20,30,40,50,60,70,80,90,100])
in_seq2 = array([15,25,35,45,55,65,75,85,95,105])
out_seq = array([in_seq1[i] + in_seq2[i] for i in range(len(in_seq1))])

print(in_seq1.shape) # (10,)
print(out_seq.shape) # (10,)

in_seq1 = in_seq1.reshape( len(in_seq1), 1)
in_seq2 = in_seq2.reshape( len(in_seq2), 1)
out_seq = out_seq.reshape( len(out_seq), 1)

print(in_seq1.shape) # (10,1) 행렬로 만들기 위해.
print(in_seq2.shape) # (10,1) 행렬로 만들기 위해.
print(out_seq.shape) # (10,1) 행렬로 만들기 위해.
print(out_seq) # (10,1) 행렬로 만들기 위해.

from numpy import hstack

dataset = hstack((in_seq1, in_seq2, out_seq))
print( dataset )
print( "=+++++++++++++++++++=" )
print( dataset.shape )
print( "=+++++++++++++++++++=" )
n_steps = 3
x , y = split_sequence3(dataset, n_steps)

for i in range(len(x)):
    print(x[i], y[i])

print(x.shape)
print(y.shape)

x = x.reshape(-1,9)
# y = y.reshape(-1,1)

print(x.shape)
print(y.shape)


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
model.add(Dense(256, activation='relu', input_shape = (9,))) # input_shape = (3,1)은  행은 제외된 3열을 1개씩 자르겠다라고 정의됨
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(8))
model.add(Dense(16))
model.add(Dense(32))
model.add(Dense(3))

model.summary()


model.compile(loss='mse', optimizer='adam', metrics=['mse']) 
model.fit(x_train, y_train, epochs = 100, batch_size = 2,
          validation_data=(x_val, y_val) ) 

# 4. 평가 예측
loss, mse = model.evaluate(x_test, y_test, batch_size = 2 )
print('mse:', mse)

# 2열이라 추가 해준다.
x_prd = np.array([[85, 90, 95],[100,105,110],[115,120,125]])
print(x_prd.shape)

x_prd = x_prd.reshape(1,9)
print(x_prd.shape)
# 열로 바꿔준다. 
# x_prd = np.transpose(x_prd)

aa = model.predict(x_prd, batch_size = 2 )
print('x_prd : ', aa)


# 실습
# 1. 함수 구조를 파악
# 2. DNN 모델을 만들어라.
# 3. 지표는 loss 
# 4. [[ 90  95]
#     [ 100  105]
#     [120 115]] 
#     predict 하세요.