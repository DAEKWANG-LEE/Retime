from keras.models import Sequential

import pandas as pd
import matplotlib.pyplot as plt

# 데이터 읽어 들이기
wine = pd.read_csv("./keras/ml/Data/winequality-white.csv", 
                        sep=";", encoding= 'utf-8'
                        )

count_data = wine.groupby('quality')["quality"].count()
print(count_data)

count_data.plot()
plt.savefig("winequality-white.png")
plt.show()
