

def __phi__(phi_0, phi_1):
    if phi_0:
        return phi_0
    return phi_1

def set_field_wrapper(base, attr, value):
    setattr(base, attr, value)
    return base

def set_index_wrapper(base, attr, value):
    setattr(base, attr, value)
    return base

def global_wrapper(x):
    return x
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from time import time
from sklearn.metrics import f1_score
_var0 = get_ipython()
_var1 = 'matplotlib'
_var2 = 'inline'
_var0.run_line_magic(_var1, _var2)
_var3 = 'student-data.csv'
student_data = pd.read_csv(_var3)
_var4 = 'Student data read successfully!'
print(_var4)
_var5 = student_data.shape
_var6 = 0
n_students = _var5[_var6]
print(n_students)
_var7 = student_data.columns
_var8 = _var7.values
_var9 = len(_var8)
_var10 = 1
n_features = (_var9 - _var10)
print(n_features)
_var11 = student_data.passed
_var12 = _var11.value_counts()
_var13 = 'yes'
n_passed = _var12[_var13]
_var14 = student_data.passed
_var15 = 'no'
_var16 = (_var14 == _var15)
n_failed = sum(_var16)
print(n_passed)
print(n_failed)
_var17 = float(n_passed)
_var18 = float(n_students)
_var19 = (_var17 / _var18)
_var20 = 100
grad_rate = (_var19 * _var20)
_var21 = 'Total number of students: {}'
_var22 = _var21.format(n_students)
print(_var22)
_var23 = 'Number of features: {}'
_var24 = _var23.format(n_features)
print(_var24)
_var25 = 'Number of students who passed: {}'
_var26 = _var25.format(n_passed)
print(_var26)
_var27 = 'Number of students who failed: {}'
_var28 = _var27.format(n_failed)
print(_var28)
_var29 = 'Graduation rate of the class: {:.2f}%'
_var30 = _var29.format(grad_rate)
print(_var30)
_var31 = student_data.columns
_var32 = (- 1)
_var33 = _var31[:_var32]
feature_cols = list(_var33)
_var34 = student_data.columns
_var35 = (- 1)
target_col = _var34[_var35]
_var36 = 'Feature columns:\n{}'
_var37 = _var36.format(feature_cols)
print(_var37)
_var38 = '\nTarget column: {}'
_var39 = _var38.format(target_col)
print(_var39)
X_all = student_data[feature_cols]
y_all = student_data[target_col]
_var40 = '\nFeature values:'
print(_var40)
_var41 = 5
_var42 = X_all.head(_var41)
print(_var42)

def preprocess_features(X):
    ' Preprocesses the student data and converts non-numeric binary variables into\n        binary (0/1) variables. Converts categorical variables into dummy variables. '
    _var43 = X.index
    output = pd.DataFrame(index=_var43)
    _var44 = X.items()
    _var45 = list(_var44)
    for _var46 in _var45:
        _var49 = 0
        col = _var46[_var49]
        _var50 = 1
        col_data = _var46[_var50]
        _var51 = col_data.dtype
        _var52 = (_var51 == object)
        if _var52:
            _var53 = ['yes', 'no']
            _var54 = [1, 0]
            col_data_0 = col_data.replace(_var53, _var54)
        col_data_1 = __phi__(col_data_0, col_data)
        _var55 = col_data_1.dtype
        _var56 = (_var55 == object)
        if _var56:
            col_data_2 = pd.get_dummies(col_data_1, prefix=col)
        col_data_3 = __phi__(col_data_2, col_data_1)
        output_0 = output.join(col_data_3)
    output_1 = __phi__(output_0, output)
    return output_1
X_all_0 = preprocess_features(X_all)
_var57 = 'Processed feature columns ({} total features):\n{}'
_var58 = X_all_0.columns
_var59 = len(_var58)
_var60 = X_all_0.columns
_var61 = list(_var60)
_var62 = _var57.format(_var59, _var61)
print(_var62)
_var63 = X_all_0.columns
nomesVar = list(_var63)
X_all_0.head()
_var64 = 0
medias0 = X_all_0.mean(axis=_var64)
_var65 = 0
desvios0 = X_all_0.std(axis=_var65)
_var66 = len(medias0)
_var67 = range(_var66)
_var68 = list(_var67)
_var69 = np.array(_var68)
_var70 = 1
eixo = (_var69 + _var70)
_var71 = 'ro'
plt.plot(eixo, medias0, _var71)
_var72 = 'media'
plt.ylabel(_var72)
_var73 = 'Media das variaveis preditivas nao padronizadas'
plt.title(_var73)
_var74 = 'ro'
plt.plot(eixo, desvios0, _var74)
_var75 = 'desvios padrao'
plt.ylabel(_var75)
_var76 = 'Desvio padrao das variaveis preditivas nao padronizadas'
plt.title(_var76)
from sklearn import preprocessing
X_all_1 = preprocessing.scale(X_all_0)
_var77 = 0
medias0_0 = X_all_1.mean(axis=_var77)
_var78 = 0
desvios0_0 = X_all_1.std(axis=_var78)
_var79 = len(medias0_0)
_var80 = range(_var79)
_var81 = list(_var80)
_var82 = np.array(_var81)
_var83 = 1
eixo_0 = (_var82 + _var83)
_var84 = 'ro'
plt.plot(eixo_0, medias0_0, _var84)
_var85 = 'media'
plt.ylabel(_var85)
_var86 = 'Media das variaveis preditivas padronizadas'
plt.title(_var86)
_var87 = 'ro'
plt.plot(eixo_0, desvios0_0, _var87)
_var88 = 'desvios padrao'
plt.ylabel(_var88)
_var89 = 'Desvio padrao das variaveis preditivas padronizadas'
plt.title(_var89)
from sklearn.model_selection import train_test_split
_var94 = 0.24
_var95 = 123
(_var90, _var91, _var92, _var93) = train_test_split(X_all_1, y_all, test_size=_var94, random_state=_var95, stratify=y_all)
X_train = _var90
X_test = _var91
y_train = _var92
y_test = _var93
_var96 = 'Training set has {} samples.'
_var97 = X_train.shape
_var98 = 0
_var99 = _var97[_var98]
_var100 = _var96.format(_var99)
print(_var100)
_var101 = 'Testing set has {} samples.'
_var102 = X_test.shape
_var103 = 0
_var104 = _var102[_var103]
_var105 = _var101.format(_var104)
print(_var105)
_var106 = 'Taxa de graduados no conjunto de treinamento: {:.2f}%'
_var107 = 100
_var108 = 'yes'
_var109 = (y_train == _var108)
_var110 = _var109.mean()
_var111 = (_var107 * _var110)
_var112 = _var106.format(_var111)
print(_var112)
_var113 = 'Taxa de graduados no conjunto de teste: {:.2f}%'
_var114 = 100
_var115 = 'yes'
_var116 = (y_test == _var115)
_var117 = _var116.mean()
_var118 = (_var114 * _var117)
_var119 = _var113.format(_var118)
print(_var119)

def train_classifier(clf, X_train_0, y_train_0):
    ' Fits a classifier to the training data. '
    start = time()
    clf_0 = clf.fit(X_train_0, y_train_0)
    end = time()
    _var120 = (end - start)
    _var121 = 3
    _var122 = round(_var120, _var121)
    return _var122

def predict_labels(clf_1, features, target):
    ' Makes predictions using a fit classifier based on F1 score. '
    start_0 = time()
    y_pred = clf_1.predict(features)
    end_0 = time()
    _var123 = (end_0 - start_0)
    _var124 = target.values
    _var125 = 'yes'
    _var126 = f1_score(_var124, y_pred, pos_label=_var125)
    _var127 = [_var123, _var126]
    return _var127

def train_predict(clf_2, X_train_1, y_train_1, X_test_0, y_test_0):
    ' Train and predict using a classifer based on F1 score. '
    tempoTreinamento = train_classifier(clf_2, X_train_1, y_train_1)
    (_var128, _var129) = predict_labels(clf_2, X_train_1, y_train_1)
    tempoPrevTreino = _var128
    prevTreino = _var129
    (_var130, _var131) = predict_labels(clf_2, X_test_0, y_test_0)
    tempoPrevTeste = _var130
    prevTeste = _var131
    _var132 = clf_2.__class__
    _var133 = _var132.__name__
    _var134 = len(X_train_1)
    _var135 = 3
    _var136 = round(tempoTreinamento, _var135)
    _var137 = 3
    _var138 = round(prevTreino, _var137)
    _var139 = 3
    _var140 = round(tempoPrevTeste, _var139)
    _var141 = 3
    _var142 = round(prevTeste, _var141)
    _var143 = [_var133, _var134, _var136, _var138, _var140, _var142]
    return _var143
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
clf_A = GaussianNB()
_var144 = 123
clf_B = LogisticRegression(random_state=_var144)
clf_C = KNeighborsClassifier()
resultado = []
scoreTrain = []
scoreTeste = []
nObs = []
nomeModelo = []
_var145 = [clf_A, clf_B, clf_C]
for modelo in _var145:
    _var146 = [100, 200, 300]
    for n in _var146:
        x = X_train[:n]
        y = y_train[:n]
        temp = train_predict(modelo, x, y, X_test, y_test_0=y_test)
        resultado.append(temp)
        _var147 = 3
        _var148 = temp[_var147]
        scoreTrain.append(_var148)
        _var149 = 5
        _var150 = temp[_var149]
        scoreTeste.append(_var150)
        _var151 = 0
        _var152 = temp[_var151]
        nomeModelo.append(_var152)
        nObs.append(n)
print(resultado)
_var153 = {'nomeModelo': nomeModelo, 'scoreTrain': scoreTrain, 'scoreTeste': scoreTeste, 'nObs': nObs}
df = pd.DataFrame(_var153)
print(df)
import seaborn as sns
_var154 = 'nObs'
_var155 = 'scoreTeste'
_var156 = 'nomeModelo'
sns.barplot(x=_var154, y=_var155, hue=_var156, data=df)
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
_var157 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
parameters = {'n_neighbors': _var157}
_var158 = X_train.shape
_var159 = 0
_var160 = _var158[_var159]
_var161 = 10
_var162 = 0.1
_var163 = 0
cv_sets = ShuffleSplit(_var160, n_iter=_var161, test_size=_var162, random_state=_var163)
clf_3 = KNeighborsClassifier()
_var164 = 'yes'
f1_scorer = make_scorer(f1_score, pos_label=_var164)
grid_obj = GridSearchCV(estimator=clf_3, param_grid=parameters, scoring=f1_scorer, cv=cv_sets)
grid_obj_0 = grid_obj.fit(X_train, y_train)
grid_obj_1 = grid_obj_0
clf_4 = grid_obj_1.best_estimator_
print(clf_4)
(_var165, _var166) = predict_labels(clf_4, X_train, y_train)
tempo = _var165
prevTreino_0 = _var166
(_var167, _var168) = predict_labels(clf_4, X_test, y_test)
tempo_0 = _var167
prevTeste_0 = _var168
_var169 = 'Tuned model has a training F1 score of {:.4f}.'
_var170 = _var169.format(prevTreino_0)
print(_var170)
_var171 = 'Tuned model has a testing F1 score of {:.4f}.'
_var172 = _var171.format(prevTeste_0)
print(_var172)
clf_default = KNeighborsClassifier()
clf_default_0 = clf_default.fit(X_train, y_train)
(_var173, _var174) = predict_labels(clf_default_0, X_train, y_train)
tempo_1 = _var173
prevTreino2 = _var174
(_var175, _var176) = predict_labels(clf_default_0, X_test, y_test)
tempo_2 = _var175
prevTeste2 = _var176
_var177 = 'Tuned model has a training F1 score of {:.4f}.'
_var178 = _var177.format(prevTreino2)
print(_var178)
_var179 = 'Tuned model has a testing F1 score of {:.4f}.'
_var180 = _var179.format(prevTeste2)
print(_var180)
_var181 = clf_4.get_params
print(_var181)
_var182 = clf_default_0.get_params
print(_var182)
