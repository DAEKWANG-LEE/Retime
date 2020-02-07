from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from keras.layers import Dense
from keras.models import Sequential

import pandas as pd
import  numpy as np
import tensorflow as tf
from keras.utils import np_utils

# 붓꽃 데이터 읽어 드리기
iris_data = pd.read_csv("./keras/ml/Data/iris.csv", 
                        encoding= 'utf-8',
                        names=['a','b','c','d','y']) #, header=None)

# 붓꽃 데이터를 레이블과 입력 데이터로 분리하기

y = iris_data.loc[:,"y"]
x = iris_data.loc[:,["a","b","c","d"]]

print(x)
print(y)

# 최종 나올 값을 문자가 아닌 숫자로 바꿔주자. 
y = y.replace('Iris-setosa', 0)
y = y.replace('Iris-versicolor', 1)
y = y.replace('Iris-virginica', 2)



x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size= 0.2, train_size=0.7, shuffle=True
)

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)   ##//원 핫 인코딩.

model = Sequential()
model.add(Dense(4, activation='relu', input_shape = (4,)))
model.add(Dense(16))  
model.add(Dense(1))
model.add(Dense(3, activation='softmax'))

model.summary()

# 3. 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc']) 
# model.compile(loss='mse', optimizer='adam', metrics=['mse']) 
# model.fit(x, y, epochs = 100, batch_size = 20)
model.fit(x_train, y_train, epochs = 100, batch_size = 4 )  # batch_size도 2배수로 정의 

        
# 평가 예측  ... 고쳐야된다.
y_pred = model.evaluate(x_test, y_test)

print(y_pred)

y_predict = model.predict(x_test)

print(" 접답? : ", y_predict )