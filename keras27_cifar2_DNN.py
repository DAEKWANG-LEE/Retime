from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, Dropout
from keras.callbacks import EarlyStopping

import numpy as np
from keras.utils import np_utils

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# print(x_train.shape)
# print(y_train.shape)


x_train = x_train.reshape(x_train.shape[0], 32, 32, 3).astype('float32')/255  #??? 255??
x_test = x_test.reshape(x_test.shape[0], 32, 32, 3).astype('float32')/255  #??? 255??

x_train = x_train.reshape(x_train.shape[0], 32*32*3)
x_test = x_test.reshape(x_test.shape[0], 32*32*3)


y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

# print("dd :  ", x_train.shape)
# print("dd :  ", y_train.shape)

model = Sequential()
model.add(Dense(512, activation='relu', input_shape = (32*32*3, )))
model.add(Dense(16))  #//// 
model.add(Dense(8))   #////
model.add(Dense(128, activation='relu')) #////, activation='relu'   8// loss: 1.5139 - accuracy: 0.4557 - val_loss: 1.5710 - val_accuracy: 0.4417
# model.add(Dropout(0.2))  #// 2dd  8//   oss: 1.5750 - accuracy: 0.4299 - val_loss: 1.5870 - val_accuracy: 0.4284
                        #  2dd 30//   loss: 1.4151 - accuracy: 0.4903 - val_loss: 1.5114 - val_accuracy: 0.4558
model.add(Dense(32))  #////  , activation='relu'  8// loss: 1.5335 - accuracy: 0.4439 - val_loss: 1.5741 - val_accuracy: 0.4475
# model.add(Dropout(0.2)) # ////        8//  loss: 1.5647 - accuracy: 0.4360 - val_loss: 1.6081 - val_accuracy: 0.4281
                        #  ///  2dd   8//  loss: 1.5785 - accuracy: 0.4277 - val_loss: 1.5978 - val_accuracy: 0.4271
                        #       2dd   8//  loss: 1.5957 - accuracy: 0.4247 - val_loss: 1.6510 - val_accuracy: 0.4208
                        #       2dd   30// loss: 1.3955 - accuracy: 0.4974 - val_loss: 1.5183 - val_accuracy: 0.4571
model.add(Dense(8)) #16
model.add(Dense(256)) #////
# model.add(Dropout(0.2))  #// 0.2  8// loss: 1.5367 - accuracy: 0.4478 - val_loss: 1.5676 - val_accuracy: 0.4409
                         #   0.5 8//  loss: 1.5759 - accuracy: 0.4332 - val_loss: 1.6096 - val_accuracy: 0.4272
                        #        8//  loss: 1.5414 - accuracy: 0.4437 - val_loss: 1.5867 - val_accuracy: 0.4324
                        #    0.1 8//  loss: 1.5284 - accuracy: 0.4482 - val_loss: 1.5551 - val_accuracy: 0.4440
                        #    0.2 30// loss: 1.3575 - accuracy: 0.5097 - val_loss: 1.5073 - val_accuracy: 0.4690
                        #   30//       loss: 1.3394 - accuracy: 0.5187 - val_loss: 1.5234 - val_accuracy: 0.4613
model.add(Dense(10, activation='softmax'))
# model.summary()



print(y_train.shape)

model.compile(loss='categorical_crossentropy',
              optimizer= 'adam',
              metrics=['accuracy'])

# 4. 평가 예측
loss, mse = model.evaluate(x_test, y_test, batch_size = 2 )
print('mse:', mse)


EarlyStopping = EarlyStopping(monitor= 'loss', patience= 20)

model.fit(x_train, y_train, validation_split= 0.2,
          epochs=100, batch_size= 32, verbose=1,
          callbacks=[EarlyStopping])

acc = model.evaluate(x_test, y_test)
