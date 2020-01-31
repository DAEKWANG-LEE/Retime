#정제된 data 를 만들어야됨. 가보자.

import numpy as np
from numpy import array
from sklearn.metrics import r2_score
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.model_selection import train_test_split

x = np.array(range(1,21))
y = np.array(range(1,21))

x = x.reshape(20,1)

print ( x.shape)
print ( y.shape)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size = 0.5, shuffle = False ### shuffle 이 추가됨.  값을 정렬시킴.
)

print("===========================")
print(x_train)
print("===========================")
print(x_test)
print("===========================")

from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

print("===========================")
print(x_train)
print("===========================")
print(x_test)
print("===========================")