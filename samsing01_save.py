import numpy as np
import pandas as pd

df1 = pd.read_csv('../Study/keras/samsung.csv', index_col=0,
                  header=0, encoding='cp949', sep=',')

print(df1)
print(df1.shape)

df2 = pd.read_csv('../Study/keras/kospi200.csv', index_col=0,
                  header=0, encoding='cp949', sep=',')

print(df2)
print(df2.shape)

# # kospi200의 모든 데이터.
# for i in range(len(df2.index)):  # 거래량 str => int 변경
#     df2.iloc[i,4] = int(df2.iloc[i,4].replace(',','')) #모든 4열의 ,는 사라지고 숫잔는 int로 바뀜.

# # 삼성전자의 모든 데이터
# for i in range(len(df1.index)):  # 거래량 str => int 변경
#     for j in range(len(df1.iloc[i])):
#         df1.iloc[i,j] = int(df1.iloc[i,4].replace(',','')) #모든 4열의 ,는 사라지고 숫잔는 int로 바뀜.
        
# print(df1)
# print(df1.shape)

# df1 = df1.sort_values(['일자'], ascending=[True])
# df2 = df2.sort_values(['일자'], ascending=[True])

# df1 = df1.values
# df2 = df2.values

# print(type(df1), type(df2))
# print(df1.shape, df2.shape)

# np.save('../Study/keras/Data/samsung.npy', arr=df1)
# np.save('../Study/keras/Data/kispi200.npy', arr=df2)