from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import  RandomForestClassifier
from sklearn.metrics import classification_report

from sklearn.model_selection import train_test_split
from keras.layers import Dense
from keras.utils import np_utils
from keras.models import Sequential

import tensorflow as tf
import pandas as pd
import numpy  as np
from keras.utils import np_utils

# 데이터 읽어 들이기
wine = pd.read_csv("./keras/ml/Data/winequality-white.csv", 
                        sep=";", encoding= 'utf-8'
                        )

# 데이터를 레이블과 데이터로 분리하기
y = wine["quality"]
x = wine.drop("quality", axis=1)

x = np.array(x)     #####////
y = np.array(y)     #####////
 
y = np_utils.to_categorical(y)    #####////

print(x.shape)
print(y.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# 2. 모델 구성
# model = RandomForestClassifier()
model = Sequential()
model.add(Dense(32, input_shape=( 11, )))
model.add(Dense(8))
model.add(Dense(16))
model.add(Dense(10, activation='softmax'))   #####////
model.summary()

# 3. 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])    #####////
# model.compile(loss='mse', optimizer='adam', metrics=['mse']) 
# model.fit(x, y, epochs = 100, batch_size = 20)
model.fit(x_train, y_train, epochs = 10, batch_size = 4 )  # batch_size도 2배수로 정의 

# 4. 평가 예측
aaa = model.evaluate(x_test,y_test)  #####////  evaluate
print(aaa)

y_pred = model.predict(x_test)
# print("정답률 : ", y_pred )

from sklearn.metrics import mean_squared_error

def RMSE(y_test, y_predcit):
    return np.sqrt(mean_squared_error(y_test, y_pred)) # lot 를 씌어주면 RMSE가 나온다.
print("RMSE : ", RMSE(y_test, y_pred))

# R2 구하기 > 1에 근사한 값이 나와야됨.
from sklearn.metrics import r2_score

r2_y_predict = r2_score(y_test, y_pred)
print("R2 : ", r2_y_predict)