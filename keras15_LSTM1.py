from numpy import array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

x = array([[1,2,3], [2,3,4], [3,4,5], [4,5,6], [5,6,7]])
y = array([4,5,6,7,8])

print(x.shape) #(5,3)
print(y.shape) #(5,)
x = x.reshape(x.shape[0], x.shape[1],1) # LSTM 을 사용하기위해 5행, 3열, 1을 자름 로 구성했음.
# x = x.reshape(5, 3, 1)
print(x.shape) #(5, 3, 1)


model = Sequential()
model.add(LSTM(32, activation='relu', input_shape = (3,1))) # input_shape = (3,1)은  행은 제외된 3열을 1개씩 자르겠다라고 정의됨
model.add(Dense(64))
model.add(Dense(256))
model.add(Dense(512))
model.add(Dense(256))
model.add(Dense(64))
model.add(Dense(128))
model.add(Dense(64))
model.add(Dense(1))

model.summary()

# 3. 훈련
model.compile(loss='mse',   # mae, mse 옵션을 사용가능.
              optimizer='adam', 
              metrics=['mae'])  # mae, acc 옵션을 사용가능. 하지만 acc는 분류모델에서만 사용해야됨.

model.fit(x, y, 
          epochs = 200, 
          batch_size = 2) 

# 4. 평가 예측
loss, mae = model.evaluate(x, y, batch_size = 2 )

print ( loss, mae )
# input data 도 reshape 으로 변환해줘야됨.
x_input = array([6,7,8])  # (3,) -> (1, 3) -> (1, 3, 1)
x_input = x_input.reshape(1,3,1) # LSTM 을 사용하기위해 1행, 3열, 1을 자름 로 구성했음.

y_predict = model.predict(x_input)
print("y_predict : ", y_predict)

'''
predict
'''