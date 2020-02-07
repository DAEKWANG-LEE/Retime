from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


import pandas as pd
from sklearn.utils.testing import all_estimators

# scikit-learn 0.20.3 에서 31개
# scikit-learn 0.21.2 에서 40개중 4개만 돔.

import warnings

warnings.filterwarnings('ignore')
iris_data = pd.read_csv("./keras/ml/Data/iris2.csv", encoding= 'utf-8' )

# 붓꽃 데이이터 레이블과 입력 데이터로 분리하기
y = iris_data.loc[:, "Name"]
x = iris_data.loc[:,[ "SepalLength","SepalWidth","PetalLength","PetalWidth"]]

# 학습 전용과 테스트 전용 분리하기
warnings.filterwarnings('ignore')
# train 과 test 합이 1로 떨어지지 않으면 error   test와 train 두개의 사이즈중 우선시 되는건 train 이다.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3,  train_size= 0.6, shuffle=True)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

model = Pipeline([("scaler", MinMaxScaler()), ('svc', SVC())]) #전처리는 minmaxscaler로 scaler로 정의하겠다. 모델은 SVC로 하겠다.

model.fit(x_train, y_train)

print("테스트 점수  : ", model.score(x_test, y_test))