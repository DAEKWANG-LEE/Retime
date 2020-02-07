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


# 데이터 읽어 들이기
wine = pd.read_csv("./keras/ml/Data/winequality-white.csv", 
                        sep=";", encoding= 'utf-8'
                        )

# 데이터를 레이블과 데이터로 분리하기
y = wine["quality"]
x = wine.drop("quality", axis=1)

print(y)
# print(x)
# y 레이블 변경하기

newlist = []
for v in list(y):
    if v <= 4:
        newlist += [0]
    elif v <= 7:
        newlist += [1]
    else:
        newlist += [2]
y = newlist
print(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# 2. 모델 구성
model = RandomForestClassifier()

# 3. 훈련
model.fit(x_train, y_train)

# 4. 평가 예측
aaa = model.score(x_test,y_test)
print(aaa)

y_pred = model. predict(x_test)
print("정답률 : ", accuracy_score(y_test, y_pred) )

print(classification_report(y_test, y_pred))