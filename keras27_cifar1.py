from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from keras.callbacks import EarlyStopping

import numpy as np
from keras.utils import np_utils

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print(x_train.shape)
print(y_train.shape)


x_train = x_train.reshape(x_train.shape[0], 32, 32, 3).astype('float32')/255  #??? 255??
x_test = x_test.reshape(x_test.shape[0], 32, 32, 3).astype('float32')/255  #??? 255??

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

print("dd :  ", x_train.shape)
print("dd :  ", y_train.shape)

model = Sequential()
model.add(Conv2D(32, kernel_size=(2, 2), strides=(1, 1), padding='same',
                 activation='relu',
                 input_shape=(32,32,3)))
model.add(Conv2D(32, (2,2)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))
model.summary()


print(y_train.shape)
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
          epochs=100, batch_size= 16, verbose=1,
          callbacks=[EarlyStopping])
acc = model.evaluate(x_test, y_test)
