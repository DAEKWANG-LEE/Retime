import numpy as np

#1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])
 
# print(x.shape)
# print(x.shape)

# 2. 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
 
model = Sequential()
 
# 초기 Dense 3개에서 다야몬드 꼴 모양으로 2진수의 배수로 크게 해주고 다시 줄여줌. 
# 모델의 dense층을 증가 시켜주니 출력값들이 출렁이지 않고 안정적임.
model.add(Dense(5, input_dim = 1))
model.add(Dense(2))
model.add(Dense(3))
model.add(Dense(1))

model.summary()
