from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso

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

# classifier 알고리즘 모두 추출하기  ----(*1)
warnings.filterwarnings('ignore')
allAlgorithms = all_estimators(type_filter="classifier") ## 리그레스, 클레시파이어.

print(allAlgorithms)
print(len(allAlgorithms))
print(type(allAlgorithms))

for(name, algorithm) in  allAlgorithms:
    # 각 알고리즘 객체 생성하기 --- (*2)
    clf = algorithm()
    
    #학습하고 평가하기 ----(*3)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print( name, "의 정답률 =  ", accuracy_score(y_test, y_pred))
 



'''  sklearn의 모델들
AdaBoostClassifier 의 정답률 =   0.9333333333333333
BaggingClassifier 의 정답률 =   0.9777777777777777
BernoulliNB 의 정답률 =   0.24444444444444444
CalibratedClassifierCV 의 정답률 =   0.9777777777777777
ComplementNB 의 정답률 =   0.5777777777777777
DecisionTreeClassifier 의 정답률 =   0.9777777777777777
ExtraTreeClassifier 의 정답률 =   0.9777777777777777
ExtraTreesClassifier 의 정답률 =   0.9777777777777777
GaussianNB 의 정답률 =   1.0
GaussianProcessClassifier 의 정답률 =   0.9777777777777777
GradientBoostingClassifier 의 정답률 =   0.9777777777777777
KNeighborsClassifier 의 정답률 =   0.9777777777777777
LabelPropagation 의 정답률 =   0.9777777777777777
LabelSpreading 의 정답률 =   0.9777777777777777
LinearDiscriminantAnalysis 의 정답률 =   0.9777777777777777
LinearSVC 의 정답률 =   0.9777777777777777
LogisticRegression 의 정답률 =   0.9777777777777777
LogisticRegressionCV 의 정답률 =   0.9333333333333333
MLPClassifier 의 정답률 =   0.9777777777777777
MultinomialNB 의 정답률 =   0.7333333333333333
NearestCentroid 의 정답률 =   0.9111111111111111
NuSVC 의 정답률 =   0.9777777777777777
PassiveAggressiveClassifier 의 정답률 =   0.5777777777777777
Perceptron 의 정답률 =   0.9111111111111111
QuadraticDiscriminantAnalysis 의 정답률 =   0.9777777777777777
RadiusNeighborsClassifier 의 정답률 =   0.9777777777777777
RandomForestClassifier 의 정답률 =   0.9777777777777777
RidgeClassifier 의 정답률 =   0.8444444444444444
RidgeClassifierCV 의 정답률 =   0.8444444444444444
SGDClassifier 의 정답률 =   0.5777777777777777
SVC 의 정답률 =   0.9777777777777777
'''