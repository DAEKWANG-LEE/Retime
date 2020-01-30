from numpy import array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM


x = array([[1,2,3,],[2,3,4],[3,4,5],[4,5,6],
            [5,6,7],[6,7,8],[7,8,9],[8,9,10],
            [9,10,11],[10,11,12],
           [20,30,40], [30,40,50],[40,50,60]])
y = array([4,5,6,7,8,9,10,11,12,13,50,60,70])

print(x.shape) #(13,3)
print(y.shape) #(13,)
x = x.reshape(x.shape[0], x.shape[1],1) # LSTM 을 사용하기위해 5행, 3열, 1을 자름 로 구성했음.
# x = x.reshape(13, 3, 1)
print(x.shape) #(13, 3, 1)


model = Sequential()
model.add(LSTM(256, activation='relu', input_shape = (3,1))) # input_shape = (3,1)은  행은 제외된 3열을 1개씩 자르겠다라고 정의됨
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(64))
model.add(Dense(8))
model.add(Dense(16))
model.add(Dense(32))
model.add(Dense(1))

# model.summary()

# 3. 훈련
model.compile(loss='mse',   # mae, mse 옵션을 사용가능.
              optimizer='adam', 
              metrics=['mae'])  # mae, acc 옵션을 사용가능. 하지만 acc는 분류모델에서만 사용해야됨.

model.fit(x, y, 
          epochs = 160, 
          batch_size = 2) 

# 4. 평가 예측
loss, mae = model.evaluate(x, y, batch_size = 2 )

print ( loss, mae )
# input data 도 reshape 으로 변환해줘야됨.
x_input = array([[6.5,7.5,8.5],[50, 60, 70],[70, 80, 90], [100,110,120]])  # (3,) -> (1, 3) -> (1, 3, 1)
x_input = x_input.reshape(4,3,1) # LSTM 을 사용하기위해 1행, 3열, 1을 자름 로 구성했음.

y_predict = model.predict(x_input)
print("y_predict : ", y_predict)