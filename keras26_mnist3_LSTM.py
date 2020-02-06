from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, LSTM
from keras.callbacks import EarlyStopping

import numpy as np
from keras.utils import np_utils

(x_train, y_train), (x_test, y_test) = mnist.load_data()


print(x_train.shape)
print(y_train.shape)

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32')/255  #??? 255??
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32')/255  #??? 255??

x_train = x_train.reshape(x_train.shape[0],28*28,1)
x_test = x_test.reshape(x_test.shape[0], 28*28,1)


y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

print("dd :  ", x_train.shape)
print("dd :  ", y_train.shape)

model = Sequential()
model.add(Reshape((16,16)))
model.add(LSTM(32, activation='relu', input_shape = (28*28,1))) # input_shape = (3,1)은  행은 제외된 3열을 1개씩 자르겠다라고 정의됨
model.add(Dense(8))
model.add(Dense(16))
model.add(Dense(32))
model.add(Dense(10, activation='softmax'))
model.summary()

# print(y_train.shape)
# print(x_train.shape)
# print(y_train.shape)
# print(type(x_train))

model.compile(loss='categorical_crossentropy',
              optimizer= 'adam',
              metrics=['accuracy'])

# 4. 평가 예측
loss, mse = model.evaluate(x_test, y_test, batch_size = 2 )
print('mse:', mse)


EarlyStopping = EarlyStopping(monitor= 'loss', patience= 20)

model.fit(x_train, y_train, validation_split= 0.2,
          epochs=10, batch_size= 20, verbose=1,
          callbacks=[EarlyStopping])
acc = model.evaluate(x_test, y_test)
