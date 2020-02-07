from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from keras.layers import Dense
from keras.models import Sequential
import  numpy as np

# 1. 데이터
x_train = [[0,0],[1,0],[0,1],[1,1]]
y_train = [0,1,1,1]

x_train = np.array(x_train)
y_train = np.array(y_train)
# y_train = y_train.reshape(-1,2)
# x_train.shape
# print(y_train.shape)

# 2. 모델
# model = LinearSVC()
# model = KNeighborsClassifier(n_neighbors = 1)
model = Sequential()
model.add(Dense(4, activation='relu', input_shape = (2,)))
model.add(Dense(16))  #//// 
model.add(Dense(1))  #//// 
# model.add(Dense(1, activation='sigmoid'))  #//// 

model.summary()

# 3. 훈련
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc']) 
model.compile(loss='mse', optimizer='adam', metrics=['mse']) 
# model.fit(x, y, epochs = 100, batch_size = 20)
model.fit(x_train, y_train, epochs = 100, batch_size = 4 )  # batch_size도 2배수로 정의 

# 4. 평가 예측
x_test = [[0,0],[1,0],[0,1],[1,1]]
x_test = np.array(x_test)

y_pred = model.evaluate(x_test, y_train)
print("skskdks ",  y_pred)

y_predict = model.predict(x_test)

print(x_test, "의 예측결과", y_predict )
# print("acc = ", accuracy_score([0,0,0,1], y_predict))