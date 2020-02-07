from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.utils.testing import all_estimators

import pandas as pd

# scikit-learn 0.20.3 에서 31개
# scikit-learn 0.21.2 에서 40개중 4개만 돔.

import warnings

warnings.filterwarnings('ignore')
iris_data = pd.read_csv("./keras/ml/Data/iris2.csv", encoding= 'utf-8' )

# 붓꽃 데이이터 레이블과 입력 데이터로 분리하기
y = iris_data.loc[:, "Name"]
x = iris_data.loc[:,[ "SepalLength","SepalWidth","PetalLength","PetalWidth"]]


# classifier 알고리즘 모두 추출하기  ----(*1)
warnings.filterwarnings('ignore')
allAlgorithms = all_estimators(type_filter="classifier") ## 리그레스, 클레시파이어.

kfold_cv = KFold(n_splits= 10, shuffle= True)  ## 4/1로 잘린다???


for(name, algorithm) in  allAlgorithms:
    # 각 알고리즘 객체 생성하기 --- (*2)
    models = algorithm()
    
    # fit이 없는데 돌아가네???
    if hasattr(models, "score"):   # score가 잇는 모델만 사용하겠다.
        #//cross_val_score 는 fit이 포함되어서 돌아감.
        scores = cross_val_score(models, x, y, cv=kfold_cv) # 모델을 가져와서, x, y를 사용할거다. 
                                                            # 검증은 kfold 로 교차 검증하겠다.
        # scores.fit()
        print(name, " 의 정답률 =  ")
        # print(scores)
        if scores <= 0.9999:
            print(scores)
        print("\n----------------------------------------------------------------------")
