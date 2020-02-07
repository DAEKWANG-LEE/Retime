from sklearn.svm import LinearSVC, SVC


from sklearn.ensemble import  RandomForestClassifier
from sklearn.metrics import classification_report

from sklearn.model_selection import train_test_split
from keras.layers import Dense
from keras.utils import np_utils
from keras.models import Sequential

import tensorflow as tf
import pandas as pd
import numpy  as np

# 데이터
from sklearn.datasets import load_boston
boston = load_boston()

x = boston.data
y = boston.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

# 모델
from sklearn.linear_model import LinearRegression, Ridge, Lasso

# model = LinearRegression()
# model = Ridge()
model = Lasso()

# 3. 훈련
model.fit(x_train, y_train)

# 4. 평가 예측
aaa = model.score(x_test,y_test)
print(" score : ", aaa)



