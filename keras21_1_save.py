from numpy import array
from keras.models import Sequential
from keras.layers import Dense, LSTM, Reshape, BatchNormalization, Dropout


x = array([[1,2,3,],[2,3,4],[3,4,5],[4,5,6],
            [5,6,7],[6,7,8],[7,8,9],[8,9,10],
            [9,10,11],[10,11,12],
           [20,30,40], [30,40,50],[40,50,60]])
y = array([4,5,6,7,8,9,10,11,12,13,50,60,70])

print(x.shape) #(13,3)
print(y.shape) #(13,)
x = x.reshape(x.shape[0], x.shape[1],1) # LSTM 을 사용하기위해 5행, 3열, 1을 자름 로 구성했음.
# x = x.reshape(13, 3, 1)
print(x.shape) #(13, 3, 1)


model = Sequential()
model.add(LSTM(256, activation='relu', input_shape = (3,1))) # input_shape = (3,1)은  행은 제외된 3열을 1개씩 자르겠다라고 정의됨
model.add(Dense(20, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(64))
model.add(Dense(32, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(8))
model.add(Dense(16, activation='relu'))
model.add(Dense(32))
model.add(Dense(1))

model.summary()
# 모델 save

model.save('./savetest01.h5')
print('저장 잘 됬다.')