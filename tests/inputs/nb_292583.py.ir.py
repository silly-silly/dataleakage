

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
from time import time
from IPython.display import display
import visuals as vs
_var0 = get_ipython()
_var1 = 'matplotlib'
_var2 = 'inline'
_var0.run_line_magic(_var1, _var2)
_var3 = 'census.csv'
data = pd.read_csv(_var3)
_var4 = 1
_var5 = data.head(n=_var4)
display(_var5)
_var6 = data.shape
_var7 = 0
n_records = _var6[_var7]
_var8 = 'income'
_var9 = data[_var8]
_var10 = '>50K'
_var11 = (_var9 == _var10)
_var12 = data[_var11]
_var13 = _var12.shape
_var14 = 0
n_greater_50k = _var13[_var14]
_var15 = 'income'
_var16 = data[_var15]
_var17 = '<=50K'
_var18 = (_var16 == _var17)
_var19 = data[_var18]
_var20 = _var19.shape
_var21 = 0
n_at_most_50k = _var20[_var21]
_var22 = (n_greater_50k / n_records)
_var23 = 100
greater_percent = (_var22 * _var23)
_var24 = 'Total number of records: {}'
_var25 = _var24.format(n_records)
print(_var25)
_var26 = 'Individuals making more than $50,000: {}'
_var27 = _var26.format(n_greater_50k)
print(_var27)
_var28 = 'Individuals making at most $50,000: {}'
_var29 = _var28.format(n_at_most_50k)
print(_var29)
_var30 = 'Percentage of individuals making more than $50,000: {:.2f}%'
_var31 = _var30.format(greater_percent)
print(_var31)
_var32 = 'income'
income_raw = data[_var32]
_var33 = 'income'
_var34 = 1
features_raw = data.drop(_var33, axis=_var34)
vs.distribution(data)
skewed = ['capital-gain', 'capital-loss']
_var35 = data[skewed]

def _func0(x):
    _var36 = 1
    _var37 = (x + _var36)
    _var38 = np.log(_var37)
    return _var38
_var39 = _var35.apply(_func0)
features_raw_0 = set_index_wrapper(features_raw, skewed, _var39)
_var40 = True
vs.distribution(features_raw_0, transformed=_var40)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
numerical = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
_var41 = data[numerical]
_var42 = scaler.fit_transform(_var41)
features_raw_1 = set_index_wrapper(features_raw_0, numerical, _var42)
_var43 = 1
_var44 = features_raw_1.head(n=_var43)
display(_var44)
features = pd.get_dummies(features_raw_1)
_var45 = '>50K'
_var46 = (income_raw == _var45)
_var47 = 'int'
income = _var46.astype(_var47)
_var48 = features.columns
encoded = list(_var48)
_var49 = '{} total features after one-hot encoding.'
_var50 = len(encoded)
_var51 = _var49.format(_var50)
print(_var51)
from sklearn.model_selection import train_test_split
_var56 = 0.2
_var57 = 0
(_var52, _var53, _var54, _var55) = train_test_split(features, income, test_size=_var56, random_state=_var57)
X_train = _var52
X_test = _var53
y_train = _var54
y_test = _var55
_var58 = 'Training set has {} samples.'
_var59 = X_train.shape
_var60 = 0
_var61 = _var59[_var60]
_var62 = _var58.format(_var61)
print(_var62)
_var63 = 'Testing set has {} samples.'
_var64 = X_test.shape
_var65 = 0
_var66 = _var64[_var65]
_var67 = _var63.format(_var66)
print(_var67)
_var68 = features.shape
_var69 = 0
_var70 = _var68[_var69]
pred = np.ones(_var70)
pred_0 = pd.Series(pred)
_var71 = (pred_0 == income)
_var72 = _var71.sum()
_var73 = len(pred_0)
accuracy = (_var72 / _var73)
sumTP = 0
_var74 = len(pred_0)
_var75 = range(_var74)
for i in _var75:
    _var76 = pred_0[i]
    _var77 = 1
    _var78 = (_var76 == _var77)
    if _var78:
        _var79 = pred_0[i]
        _var80 = income[i]
        _var81 = (_var79 == _var80)
        if _var81:
            _var82 = 1
            sumTP_0 = (sumTP + _var82)
        sumTP_1 = __phi__(sumTP_0, sumTP)
    sumTP_2 = __phi__(sumTP_1, sumTP)
sumTP_3 = __phi__(sumTP_2, sumTP)
_var83 = 1
_var84 = (pred_0 == _var83)
_var85 = _var84.sum()
precision = (sumTP_3 / _var85)
_var86 = 1
_var87 = (income == _var86)
_var88 = _var87.sum()
recall = (sumTP_3 / _var88)
_var89 = 1
_var90 = 0.5
_var91 = 2
_var92 = (_var90 ** _var91)
_var93 = (_var89 + _var92)
_var94 = (_var93 * precision)
_var95 = (_var94 * recall)
_var96 = 0.5
_var97 = 2
_var98 = (_var96 ** _var97)
_var99 = (_var98 * precision)
_var100 = (_var99 + recall)
fscore = (_var95 / _var100)
_var101 = 'Naive Predictor: [Accuracy score: {:.4f}, F-score: {:.4f}]'
_var102 = _var101.format(accuracy, fscore)
print(_var102)
from sklearn.metrics import fbeta_score, accuracy_score

def train_predict(learner, sample_size, X_train_0, y_train_0, X_test_0, y_test_0):
    '\n    inputs:\n       - learner: the learning algorithm to be trained and predicted on\n       - sample_size: the size of samples (number) to be drawn from training set\n       - X_train: features training set\n       - y_train: income training set\n       - X_test: features testing set\n       - y_test: income testing set\n    '
    results = {}
    train_data = X_train_0[:sample_size]
    train_label = y_train_0[:sample_size]
    start = time()
    learner_0 = learner.fit(train_data, train_label)
    learner_1 = learner_0
    end = time()
    _var103 = 'train_time'
    _var104 = (end - start)
    results_0 = set_index_wrapper(results, _var103, _var104)
    start_0 = time()
    predictions_test = learner_1.predict(X_test_0)
    _var105 = 300
    _var106 = X_train_0[:_var105]
    predictions_train = learner_1.predict(_var106)
    end_0 = time()
    _var107 = 'pred_time'
    _var108 = (end_0 - start_0)
    results_1 = set_index_wrapper(results_0, _var107, _var108)
    _var109 = 'acc_train'
    _var110 = 300
    _var111 = y_train_0[:_var110]
    _var112 = accuracy_score(_var111, predictions_train)
    results_2 = set_index_wrapper(results_1, _var109, _var112)
    _var113 = 'acc_test'
    _var114 = accuracy_score(y_test_0, predictions_test)
    results_3 = set_index_wrapper(results_2, _var113, _var114)
    _var115 = 'f_train'
    _var116 = 300
    _var117 = y_train_0[:_var116]
    _var118 = 0.5
    _var119 = fbeta_score(_var117, predictions_train, beta=_var118)
    results_4 = set_index_wrapper(results_3, _var115, _var119)
    _var120 = 'f_test'
    _var121 = 0.5
    _var122 = fbeta_score(y_test_0, predictions_test, beta=_var121)
    results_5 = set_index_wrapper(results_4, _var120, _var122)
    _var123 = '{} trained on {} samples.'
    _var124 = learner_1.__class__
    _var125 = _var124.__name__
    _var126 = _var123.format(_var125, sample_size)
    print(_var126)
    return results_5
from sklearn.ensemble import AdaBoostClassifier
from sklearn import svm
from sklearn import linear_model
_var127 = 0
clf_A = AdaBoostClassifier(random_state=_var127)
_var128 = 1
clf_B = svm.SVC(random_state=_var128)
_var129 = 2
clf_C = linear_model.LogisticRegression(random_state=_var129)
_var130 = X_train.shape
_var131 = 0
_var132 = _var130[_var131]
_var133 = 0.01
_var134 = (_var132 * _var133)
samples_1 = int(_var134)
_var135 = X_train.shape
_var136 = 0
_var137 = _var135[_var136]
_var138 = 0.1
_var139 = (_var137 * _var138)
samples_10 = int(_var139)
_var140 = X_train.shape
_var141 = 0
_var142 = _var140[_var141]
samples_100 = int(_var142)
results_6 = {}
_var143 = [clf_A, clf_B, clf_C]
for clf in _var143:
    _var144 = clf.__class__
    clf_name = _var144.__name__
    _var145 = {}
    results_7 = set_index_wrapper(results_6, clf_name, _var145)
    _var146 = [samples_1, samples_10, samples_100]
    _var147 = enumerate(_var146)
    for _var148 in _var147:
        _var151 = 0
        i_0 = _var148[_var151]
        _var152 = 1
        samples = _var148[_var152]
        _var153 = results_7[clf_name]
        _var154 = train_predict(clf, samples, X_train, y_train, X_test, y_test)
        _var153_0 = set_index_wrapper(_var153, i_0, _var154)
    i_1 = __phi__(i_0, i)
results_8 = __phi__(results_7, results_6)
i_2 = __phi__(i_1, i)
vs.evaluate(results_8, accuracy, fscore)
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
_var155 = 0
clf_0 = AdaBoostClassifier(random_state=_var155)
_var156 = [10, 100, 500]
_var157 = [0.2, 0.6, 1.0]
parameters = {'n_estimators': _var156, 'learning_rate': _var157}
_var158 = 0.5
scorer = make_scorer(fbeta_score, beta=_var158)
grid_obj = GridSearchCV(clf_0, parameters)
grid_fit = grid_obj.fit(X_train, y_train)
grid_obj_0 = grid_fit
best_clf = grid_obj_0.best_estimator_
_var159 = clf_0.fit(X_train, y_train)
predictions = _var159.predict(X_test)
best_predictions = best_clf.predict(X_test)
_var160 = 'Unoptimized model\n------'
print(_var160)
_var161 = 'Accuracy score on testing data: {:.4f}'
_var162 = accuracy_score(y_test, predictions)
_var163 = _var161.format(_var162)
print(_var163)
_var164 = 'F-score on testing data: {:.4f}'
_var165 = 0.5
_var166 = fbeta_score(y_test, predictions, beta=_var165)
_var167 = _var164.format(_var166)
print(_var167)
_var168 = '\nOptimized Model\n------'
print(_var168)
_var169 = 'Final accuracy score on the testing data: {:.4f}'
_var170 = accuracy_score(y_test, best_predictions)
_var171 = _var169.format(_var170)
print(_var171)
_var172 = 'Final F-score on the testing data: {:.4f}'
_var173 = 0.5
_var174 = fbeta_score(y_test, best_predictions, beta=_var173)
_var175 = _var172.format(_var174)
print(_var175)
model = best_clf
importances = model.feature_importances_
vs.feature_plot(importances, X_train, y_train)
from sklearn.base import clone
_var176 = X_train.columns
_var177 = _var176.values
_var178 = np.argsort(importances)
_var179 = (- 1)
_var180 = _var178[::_var179]
_var181 = 5
_var182 = _var180[:_var181]
_var183 = _var177[_var182]
X_train_reduced = X_train[_var183]
_var184 = X_test.columns
_var185 = _var184.values
_var186 = np.argsort(importances)
_var187 = (- 1)
_var188 = _var186[::_var187]
_var189 = 5
_var190 = _var188[:_var189]
_var191 = _var185[_var190]
X_test_reduced = X_test[_var191]
_var192 = clone(best_clf)
clf_1 = _var192.fit(X_train_reduced, y_train)
_var192_0 = clf_1
reduced_predictions = clf_1.predict(X_test_reduced)
_var193 = 'Final Model trained on full data\n------'
print(_var193)
_var194 = 'Accuracy on testing data: {:.4f}'
_var195 = accuracy_score(y_test, best_predictions)
_var196 = _var194.format(_var195)
print(_var196)
_var197 = 'F-score on testing data: {:.4f}'
_var198 = 0.5
_var199 = fbeta_score(y_test, best_predictions, beta=_var198)
_var200 = _var197.format(_var199)
print(_var200)
_var201 = '\nFinal Model trained on reduced data\n------'
print(_var201)
_var202 = 'Accuracy on testing data: {:.4f}'
_var203 = accuracy_score(y_test, reduced_predictions)
_var204 = _var202.format(_var203)
print(_var204)
_var205 = 'F-score on testing data: {:.4f}'
_var206 = 0.5
_var207 = fbeta_score(y_test, reduced_predictions, beta=_var206)
_var208 = _var205.format(_var207)
print(_var208)
