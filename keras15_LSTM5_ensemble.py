from numpy import array
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, Input


x1 = array([[1,2,3,],[2,3,4],[3,4,5],[4,5,6],
            [5,6,7],[6,7,8],[7,8,9],[8,9,10],
            [9,10,11],[10,11,12],
           [20,30,40], [30,40,50],[40,50,60]])
y1 = array([4,5,6,7,8,9,10,11,12,13,50,60,70])

x2 = array([[10,20,30],[20,30,40],[30,40,50],[40,50,60],
            [50,60,70],[60,70,80],[70,80,90],[80,90,100],
            [90,100,110],[100,110,120],
           [2,3,4], [3,4,5],[4,5,6]])
y2 = array([40,50,60,70,80,90,100,110,120,130,5,6,7])

print(x1.shape) #(13,3)
print(y1.shape) #(13,)
x1 = x1.reshape(x1.shape[0], x1.shape[1],1) # LSTM 을 사용하기위해 5행, 3열, 1을 자름 로 구성했음.
# x = x.reshape(13, 3, 1)
print(x1.shape) #(13, 3, 1)
x2 = x2.reshape(x2.shape[0], x2.shape[1],1) # LSTM 을 사용하기위해 5행, 3열, 1을 자름 로 구성했음.
# x = x.reshape(13, 3, 1)
print(x2.shape) #(13, 3, 1)


from sklearn.model_selection import train_test_split
#파라미터를 늘여보자.
x1_train, x1_test, x2_train, x2_test, y1_train, y1_test  = train_test_split(
    x1, x2, y1, train_size = 0.6, random_state = 66, # random_state 66번째 의 랜덤스테이트 모듈을 사용하겠다(랜덤 난수 사용).
    shuffle = False
)

x1_val, x1_test, x2_val, x2_test, y_val, y_test = train_test_split(
    x1_test, x2_test, y1_test, test_size = 0.4, random_state = 66, # random_state 66번째 의 랜덤스테이트 모듈을 사용하겠다(랜덤 난수 사용).
    shuffle = False
)


input1 = Input(shape=(3,1))
dense1 = LSTM(256, activation='relu')(input1)# input_shape = (3,1)은  행은 제외된 3열을 1개씩 자르겠다라고 정의됨
dense1 = Dense(32, activation='relu')(dense1)
dense1 = Dense(64)(dense1)
dense1 = Dense(8)(dense1)
dense1 = Dense(64)(dense1)
dense1 = Dense(16, activation='relu')(dense1)
dense1 = Dense(32)(dense1)
dense1 = Dense(1)(dense1)

input2 = Input(shape=(3,1))
dense2 = LSTM(256, activation='relu')(input2)
dense2 = Dense(32, activation='relu')(dense2)
dense2 = Dense(64)(dense2)
dense2 = Dense(8)(dense2)
dense2 = Dense(64)(dense2)
dense2 = Dense(16, activation='relu')(dense2)
dense2 = Dense(32)(dense2)
dense2 = Dense(1)(dense2)

# 모델을 합치자!
from tensorflow.python.keras.layers.merge import concatenate
merge1 = concatenate([output1, output2])

# hedden을 또 만들자 즉 아래에 모델을 또 만들었다. 

middle1 = Dense(4)(merge1)
middle2 = Dense(7)(middle1)
output = Dense(1)(middle2)

#함수형 모델은 명시를 따로 아래에 해준다.
model = Model(inputs = [input1, input2], outputs = output)

model.summary()

# 3. 훈련
model.compile(loss='mse',   # mae, mse 옵션을 사용가능.
              optimizer='adam', 
              metrics=['mae'])  # mae, acc 옵션을 사용가능. 하지만 acc는 분류모델에서만 사용해야됨.

from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor ='loss', # loss를 모니터 링하겠다.
                               patience = 12, # 내가 원하는 loss값을 20번 확인하겠다.
                               mode = 'min') # loss를 보면 min으로 준다. 'auto'는 자동
                                # 최대값을 찍고 그 max나 min이 20번 이상 개선이 없을시.

model.fit(x, y, 
          epochs = 500, 
          batch_size = 2,
          verbose = 1,
          callbacks=[early_stopping] #위에 정의한 early_stopping을 fit에다가 정의한다.
          ) # verbose 훈련의 과정을 출력함. 0 1개만 1 전부다 2 정보 추림

# 4. 평가 예측
loss, mae = model.evaluate(x, y, batch_size = 2 )

print ( loss, mae )
# input data 도 reshape 으로 변환해줘야됨.
x_input = array([[6.5,7.5,8.5],[50, 60, 70],[70, 80, 90], [100,110,120]])  # (3,) -> (1, 3) -> (1, 3, 1)
x_input = x_input.reshape(4,3,1) # LSTM 을 사용하기위해 1행, 3열, 1을 자름 로 구성했음.

y_predict = model.predict(x_input)
print("y_predict : ", y_predict)