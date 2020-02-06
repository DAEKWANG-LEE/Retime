from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten

model = Sequential()
model.add(Conv2D(7, (2,2),strides=2 padding= 'valid', #valid
                 input_shape=(5, 5, 1))) # , strides 의 defult는 1 또는 (1,1 ) 이다.

model.add(Conv2D(100, (2,2)))
model.add(MaxPooling2D((2,2))) # , strides 의 defult는 입력값을 따라간다.
model.add(Flatten())  # 1차원으로 바꿔준다. 
model.add(Dense(1))

model.summary()