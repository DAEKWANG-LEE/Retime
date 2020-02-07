from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

from keras.layers import Dense
from keras.models import Sequential

from numpy import array
import tensorflow as tf
import pandas as pd
import numpy  as np

# seed 값 생성
seed = 0
np.random.seed(seed)
tf.set_random_seed(seed)

# 데이터 로드
dataset = np.loadtxt("./keras/ml/pima-indians-diabetes.csv", delimiter = ",")
X = np.array(dataset[:,0:8])
Y = np.array(dataset[:,8])

print(X.shape)
print(Y.shape)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    X, Y, train_size = 0.6, random_state = 66, # random_state 66번째 의 랜덤스테이트 모듈을 사용하겠다(랜덤 난수 사용).
    shuffle = False
)


# 모델의 설정
model = Sequential
model.add(Dense(12, activation = 'relu', input_shape= 8 ))
model.add(Dense(8, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))

# 모델 컴파일
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics= ['accuracy'])

#모델 실행
model.fit(_train,y_train, epochs=200, batch_size= 10)


# 평가 예측  ... 고쳐야된다.
y_pred = model.evaluate(x_test, y_test)
# model.fit(x_train,y_train)

y_predict = model.predict(x_test)
print("아브라 카타브라 :", y_predict)

# 출력
# print("\n Accuracy: %.4f" % (model.evaluate(X,Y)[1]))