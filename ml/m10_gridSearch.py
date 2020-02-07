from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVC

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
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

# 그리드 서치에서 사용 할 매개 변수 -- (*1)

parameters = [
    {"C": [1,10,100,1000], "kernel":["linear"]},
    {"C": [1,10,100,1000], "kernel":["rbf"], "gamma":[0.001, 0.0001]},
    {"C": [1,10,100,1000], "kernel":["sigmoid"], "gamma": [0.001, 0.0001]}
]

# 그리드 서치 --- (*2)
kfold_cv = KFold(n_splits= 5, shuffle=True)
model = GridSearchCV( SVC(), parameters, cv=kfold_cv)
model.fit(x_train, y_train)
print("/n-------------------")
print(" 최적의 매개 변수 = ", model.best_estimator_)

# 최적의 매개 변수로 평가하기 ---(*3)
y_pred = model.predict(x_test)
print("/n-------------------")
print("최종 정답률 =  ", accuracy_score(y_test, y_pred))