from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from keras.callbacks import EarlyStopping

import numpy as np
from keras.utils import np_utils

(x_train, y_train), (x_test, y_test) = mnist.load_data()


print(x_train.shape)
print(y_train.shape)

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32')/255  #??? 255??
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32')/255  #??? 255??

x_train = x_train.reshape(x_train.shape[0], 28*28 )
x_test = x_test.reshape(x_test.shape[0], 28*28 )

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

print("dd :  ", x_train.shape)
print("dd :  ", y_train.shape)

model = Sequential()
model.add(Dense(32, activation='relu', input_shape=(28*28, )))
# model.add(Conv2D(32, (2,2)))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Flatten())
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

hist = model.fit(x_train, y_train, validation_split= 0.2,
                epochs=10, batch_size= 8, verbose=1,
                callbacks=[EarlyStopping]
                
                )

# model.evaluate(x_test, y_test)

import matplotlib.pyplot as plt

print(hist.history.keys())

plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_accuracy'])
plt.title('model loss, accuracy')
plt.ylabel('loss, acc')
plt.xlabel(['epoch'])
plt.legend(['train loss', 'test loss', 'train acc', 'test acc'])
plt.show()


# acc = model.evaluate(x_test, y_test)
