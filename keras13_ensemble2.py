#1. 데이터
import numpy as np

# 백터에서 행렬로 바뀜.
x1 = np.array([range(1, 101), range(101, 201), range(301,401)])
x2 = np.array([range(1001, 1101), range(1101, 1201), range(1301,1401)])

y1 = np.array([range(1, 101), range(101, 201), range(301,401)])
y2 = np.array([range(1001, 1101), range(1101, 1201), range(1301,1401)])
y3 = np.array([range(1, 101), range(101, 201), range(301,401)])


print(x1.shape) #(3, 100)
print(y1.shape) #(1, 200)
print(y2.shape) #(200,)

print(x2.shape) #(3, 100)
print(y2.shape) #(1, 200)


# reshape 말고 transpose 로 해주자.
x1 = np.transpose(x1)
x2 = np.transpose(x2)

y1 = np.transpose(y1)
y2 = np.transpose(y2) # 바
y3 = np.transpose(y3)

# print(x1.shape) 
# print(y1.shape)

from sklearn.model_selection import train_test_split


#파라미터를 늘여보자.
x1_train, x1_test, x2_train, x2_test, y1_train, y1_test  = train_test_split(
    x1, x2, y1, train_size = 0.6, random_state = 66, # random_state 66번째 의 랜덤스테이트 모듈을 사용하겠다(랜덤 난수 사용).
    shuffle = False
)

x1_val, x1_test, x2_val, x2_test, y1_val, y1_test = train_test_split(
    x1_test, x2_test, y1_test, test_size = 0.5, random_state = 66, # random_state 66번째 의 랜덤스테이트 모듈을 사용하겠다(랜덤 난수 사용).
    shuffle = False
)

# 위에 붙여 써도 되지만  따로 하나 나누겠음.
y2_train, y2_test, y3_train, y3_test = train_test_split(  # 앞에 y이기때문에 아래의 y2,y3로 나와야됨.
    y2, y3, train_size = 0.6, random_state = 66, # random_state 66번째 의 랜덤스테이트 모듈을 사용하겠다(랜덤 난수 사용).
    shuffle = False
)

y2_val, y2_test, y3_val, y3_test = train_test_split(  # 앞에 y2_val이기때문에 아래의 y2_test, y3_test로 나와야됨.
    y2_test, y3_test, test_size = 0.5, 
    random_state = 66, # random_state 66번째 의 랜덤스테이트 모듈을 사용하겠다(랜덤 난수 사용).
    shuffle = False
)

print(y3_train.shape)
print(y3_test.shape)
print(y3_val.shape)

# 2. 모델 구성
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
 
# model = Sequential()

input1 = Input(shape=(3,))
dense1 = Dense(5)(input1)
dense2 = Dense(2)(dense1)
dense3 = Dense(3)(dense2)
output1 = Dense(11)(dense3)

input2= Input(shape=(3,))
dense21 = Dense(7)(input2)
output2 = Dense(4)(dense21)
# ? output 이 5인게 가능한가? >>> concatenate 를 사용하면 output 레이어는 hedden 레이어다. 아직 y가 나오지않으면 hedden이다.

# 모델을 합치자!
# tensorflow/tensorflow/python/keras/layers/merge.py /  순이다.
from tensorflow.python.keras.layers.merge import concatenate
merge1 = concatenate([output1, output2])

# hedden을 또 만들자 즉 아래에 모델을 또 만들었다. 
middle1 = Dense(4)(merge1)
middle2 = Dense(7)(middle1)
middle3 = Dense(11)(middle2) # 현재 merge된 마지막 레이어 

#분개 시키자.
#레이어를 정의 해주자.
output_1 = Dense(30)(middle3) # 1번째 아웃풋 모델
output_1 = Dense(3)(output_1) # !!output 마지막 column은 동일하기 3개

output_2 = Dense(300)(middle3) # 2번째 아웃풋 모델
output_2 = Dense(6)(output_2)
output_2 = Dense(3)(output_2) # !!output 마지막 column은 동일하기 3개

output_3 = Dense(10)(middle3) # 3번째 아웃풋 모델
output_3 = Dense(3)(output_3) # !!output 마지막 column은 동일하기 3개

#함수형 모델은 명시를 따로 아래에 해준다.
model = Model(inputs = [input1, input2], 
              outputs = [output_1, output_2, output_3])
 
model.summary()


# 3. 훈련
model.compile(loss='mse',   # mae, mse 옵션을 사용가능.
              optimizer='adam', 
              metrics=['mae'])  # mae, acc 옵션을 사용가능. 하지만 acc는 분류모델에서만 사용해야됨.
model.fit([x1_train, x2_train], [y1_train, y2_train, y3_train], 
          epochs = 100, batch_size = 2,
          validation_data = ([x1_val, x2_val], [y1_val, y2_val, y3_val])) 

# 4. 평가 예측
aaa = model.evaluate([x1_test, x2_test], [y1_test, y2_test, y3_test], batch_size = 2 )

# 1. 변수를 1개
# 2. 변수를 mse 갯수별로

print('aaa:', aaa)

# 2열이라 추가 해준다.  // 입력단이 2개라 2가지의 prd를 만들어준다.
x1_prd = np.array([[201,202,203],[204, 205, 206],[301,302,303]])
x2_prd = np.array([[201,202,203],[204, 205, 206],[301,302,303]]) ##행은 무시. 
# 열로 바꿔준다.  // 위와 동일하게 추가해준다.
x1_prd = np.transpose(x1_prd)
x2_prd = np.transpose(x2_prd)

# 입력단에 2개라 list로 만들어준다.
predict_aa = model.predict([x1_prd, x2_prd], batch_size = 2 )
print('predict_aa : ', predict_aa)

#  y_predict 만들어 주자.  / list 를 추가해주자.
#  y1_predict[0] model.predict 
y1_predict = model.predict([x1_test, x2_test], batch_size = 2 )
print('y1_predict : ', y1_predict)
print('y1_predict shape : ', np.array(y1_predict).shape) # 리스트를 shape 하려면 np.array(y1_predict).shape
print('y1_predict[0] : ', y1_predict[0])


# #RMSE 구하기
from sklearn.metrics import mean_squared_error

def RMSE(y1_test, y1_predict):
    return np.sqrt(mean_squared_error(y1_test, y1_predict)) # lot 를 씌어주면 RMSE가 나온다.

# print("RMSE : ", RMSE([y_test], y1_predict))
rm1 = RMSE(y1_predict[0], y1_test)
rm2 = RMSE(y1_predict[1], y2_test)
rm3 = RMSE(y1_predict[2], y3_test)

print( rm1, rm2, rm3)

RMSE = (rm1 + rm2 + rm3) / 3

print("RMSE : ", RMSE )

# R2 구하기 > 1에 근사한 값이 나와야됨.
from sklearn.metrics import r2_score

r2_y_predict1 = r2_score(y1_test, y1_predict[0])
r2_y_predict2 = r2_score(y2_test, y1_predict[1])
r2_y_predict3 = r2_score(y3_test, y1_predict[2])

r2_y_predict = (r2_y_predict1+r2_y_predict2+r2_y_predict3) / 3

print( r2_y_predict1, r2_y_predict2, r2_y_predict3)
print("R2 : ", r2_y_predict)
# 돌릴때 우선 Total params:  수치를 자료의 갯수보다 많은면 불필요하다. 왜? 자료가 200개 이상인데 개산이 2번 이상이 효율이 적음.
