#1. 데이터
import numpy as np

# 백터에서 행렬로 바뀜.
x = np.array([range(1, 101), range(101, 201), range(301,401)])
y = np.array([range(1, 101)])
y2 = np.array(range(1, 201))
print(x.shape) #(3, 100)
print(y.shape) #(1, 200)
print(y2.shape) #(200,)


# reshape 말고 transpose 로 해주자.
x = np.transpose(x)
y = np.transpose(y)

print(x.shape) 
print(y.shape)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size = 0.6, random_state = 66, # random_state 66번째 의 랜덤스테이트 모듈을 사용하겠다(랜덤 난수 사용).
    shuffle = False
)

x_val, x_test, y_val, y_test = train_test_split(
    x_test, y_test, test_size = 0.5, random_state = 66, # random_state 66번째 의 랜덤스테이트 모듈을 사용하겠다(랜덤 난수 사용).
    shuffle = False
)

# 2. 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
 
model = Sequential()
 
# 나쁜 모델로 만들어라.
# 실습 r2 0.5 이하 만들기
# epoch 100이상
# 노드 총합 50 이상
# 회든레이어 3개이상
# 과적합을 하게 되면 나쁘모델이 나온다 즉 레이어나 노드를 너무 많이 주면 나쁜모델이 된다.
model.add(Dense(6, input_shape = (3,)))
model.add(Dense(12))
model.add(Dense(18))
model.add(Dense(18))
model.add(Dense(12))
# 가 1개이기 때문에 output도 1로 한다.
model.add(Dense(1))

model.summary()

# 3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse']) 
model.fit(x_train, y_train, epochs = 100, batch_size = 50,
          validation_data=(x_val, y_val) ) 

# 4. 평가 예측
loss, mse = model.evaluate(x_test, y_test, batch_size = 50 )
print('mse:', mse)

# 2열이라 추가 해준다.
x_prd = np.array([[201,202,203],[204, 205, 206],[301,302,303]])
# 열로 바꿔준다. 
x_prd = np.transpose(x_prd)

aa = model.predict(x_prd, batch_size = 50 )
print('x_prd : ', aa)

# bb = model.predict(x_test,  batch_size = 2 )
# print('x : ', bb)



#  y_predict 만들어 주자.
y_predict = model.predict(x_test, batch_size = 50 )
#RMSE 구하기
from sklearn.metrics import mean_squared_error

def RMSE(y_test, y_predcit):
    return np.sqrt(mean_squared_error(y_test, y_predict)) # lot 를 씌어주면 RMSE가 나온다.
print("RMSE : ", RMSE(y_test, y_predict))

# R2 구하기 > 1에 근사한 값이 나와야됨.
from sklearn.metrics import r2_score

r2_y_predict = r2_score(y_test, y_predict)
print("R2 : ", r2_y_predict)

# 돌릴때 우선 Total params:  수치를 자료의 갯수보다 많은면 불필요하다. 왜? 자료가 200개 이상인데 개산이 2번 이상이 효율이 적음.
