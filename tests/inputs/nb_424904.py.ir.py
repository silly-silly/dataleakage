

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
import pandas as pd
import numpy as np
from io import StringIO
csv_data = 'A,B,C,D\n1.0,2.0,3.0,4.0\n5.0,6.0,,8.0\n0.0,11.0,12.0,'
csv_data_0 = str(csv_data)
_var0 = StringIO(csv_data_0)
df = pd.read_csv(_var0)
df.dropna()
_var1 = 1
df.dropna(axis=_var1)
from sklearn.impute import SimpleImputer as Imputer
_var2 = 'NaN'
_var3 = 'mean'
_var4 = 0
imr = Imputer(missing_values=_var2, strategy=_var3, axis=_var4)
imr_0 = imr.fit(df)
imr_1 = imr_0
_var5 = df.values
imputed_data = imr_1.transform(_var5)
imputed_data
'\nHandling categoical data\n'
_var6 = ['green', 'M', 10.1, 'class1']
_var7 = ['red', 'L', 13.5, 'class2']
_var8 = ['blue', 'XL', 15.3, 'class1']
_var9 = [_var6, _var7, _var8]
df_0 = pd.DataFrame(_var9)
_var10 = ['color', 'size', 'price', 'classlabel']
df_1 = set_field_wrapper(df_0, 'columns', _var10)
size_mapping = {'XL': 3, 'L': 2, 'M': 1}
_var11 = 'size'
_var12 = 'size'
_var13 = df_1[_var12]
_var14 = _var13.map(size_mapping)
df_2 = set_index_wrapper(df_1, _var11, _var14)
class_mapping = {label: idx for (idx, label) in enumerate(np.unique(df['classlabel']))}
_var15 = 'classlabel'
_var16 = 'classlabel'
_var17 = df_2[_var16]
_var18 = _var17.map(class_mapping)
df_3 = set_index_wrapper(df_2, _var15, _var18)
_var19 = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data'
_var20 = None
df_wine = pd.read_csv(_var19, header=_var20)
_var21 = ['Class label', 'Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium', 'Total phenols', 'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins', 'Color intensity', 'Hue', 'OD280/OD315 of diluted wines', 'Proline']
df_wine_0 = set_field_wrapper(df_wine, 'columns', _var21)
from sklearn.model_selection import train_test_split
_var22 = df_wine_0.iloc
_var23 = 1
_var24 = _var22[:, _var23:]
_var25 = _var24.values
_var26 = df_wine_0.iloc
_var27 = 0
_var28 = _var26[:, _var27]
_var29 = _var28.values
X = _var25
y = _var29
_var34 = 0.3
_var35 = 0
(_var30, _var31, _var32, _var33) = train_test_split(X, y, test_size=_var34, random_state=_var35)
X_train = _var30
X_test = _var31
y_train = _var32
y_test = _var33
'\nFeatures prepocessing includes two main methods\nOne is normalization, another is standardization\n'
from sklearn.preprocessing import MinMaxScaler
mms = MinMaxScaler()
X_train_norm = mms.fit_transform(X_train)
X_test_norm = mms.transform(X_test)
from sklearn.preprocessing import StandardScaler
std = StandardScaler()
X_train_std = std.fit_transform(X_train)
X_test_std = std.transform(X_test)
'Selecting meaningful features\nIn the overfitting case, there are 4 commmon ways\n1. More training data\n2. penalty\n3. simpler model\n4. reduce dimensionality for data\n'
from sklearn.linear_model import LogisticRegression
_var36 = 'l1'
_var37 = 0.1
lr = LogisticRegression(penalty=_var36, C=_var37)
lr_0 = lr.fit(X_train_std, y_train)
Training_accuracy = lr_0.score(X_train_std, y_train)
Testing_accuracy = lr_0.score(X_test_std, y_test)
import matplotlib.pyplot as plt
fig = plt.figure()
_var38 = 111
ax = plt.subplot(_var38)
colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'pink', 'lightgreen', 'lightblue', 'gray', 'indigo', 'orange']
_var39 = []
_var40 = []
weights = _var39
params = _var40
_var41 = (- 4)
_var42 = 6
_var43 = np.arange(_var41, _var42)
for c in _var43:
    _var44 = 'l1'
    _var45 = 10
    _var46 = (_var45 ** c)
    _var47 = 0
    lr_1 = LogisticRegression(penalty=_var44, C=_var46, random_state=_var47)
    lr_2 = lr_1.fit(X_train_std, y_train)
    _var48 = lr_2.coef_
    _var49 = 1
    _var50 = _var48[_var49]
    weights.append(_var50)
    _var51 = 10
    _var52 = (_var51 ** c)
    params.append(_var52)
lr_3 = __phi__(lr_2, lr_0)
weights_0 = np.array(weights)
_var53 = weights_0.shape
_var54 = 1
_var55 = _var53[_var54]
_var56 = range(_var55)
_var57 = list(_var56)
_var58 = zip(_var57, colors)
for _var59 in _var58:
    _var62 = 0
    column = _var59[_var62]
    _var63 = 1
    color = _var59[_var63]
    _var64 = weights_0[:, column]
    _var65 = df_wine_0.columns
    _var66 = 1
    _var67 = (column + _var66)
    _var68 = _var65[_var67]
    plt.plot(params, _var64, label=_var68, color=color)
_var69 = 0
_var70 = 'black'
_var71 = '--'
_var72 = 3
plt.axhline(_var69, color=_var70, linestyle=_var71, linewidth=_var72)
_var73 = 10
_var74 = (- 5)
_var75 = (_var73 ** _var74)
_var76 = 10
_var77 = 5
_var78 = (_var76 ** _var77)
_var79 = [_var75, _var78]
plt.xlim(_var79)
_var80 = 'weight coefficient'
plt.ylabel(_var80)
_var81 = 'C'
plt.xlabel(_var81)
_var82 = 'log'
plt.xscale(_var82)
_var83 = 'upper left'
plt.legend(loc=_var83)
_var84 = 'upper center'
_var85 = 1.38
_var86 = 1.03
_var87 = (_var85, _var86)
_var88 = 1
_var89 = True
ax.legend(loc=_var84, bbox_to_anchor=_var87, ncol=_var88, fancybox=_var89)
plt.show()
'\n@Author: Darcy\n@Date: May, 17, 2017\n@Topic: SBS\nA classic sequential feature selection algorithm is Sequential Backward Selection (SBS)\nwhich aims to reduce the dimensionality of the initial feature subspace \nwith a minimum decay in performance of the classifier \nto improve upon computational efficiency\n'
from sklearn.base import clone
from itertools import combinations
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
'\n@Author: Darcy\n@Date: May, 17, 2017\n@Topic: SBS\nA classic sequential feature selection algorithm is Sequential Backward Selection (SBS)\nwhich aims to reduce the dimensionality of the initial feature subspace \nwith a minimum decay in performance of the classifier \nto improve upon computational efficiency\n'
from sklearn.base import clone
from itertools import combinations
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class SBS():

    def __init__(self, estimator, k_features, scoring=accuracy_score, test_size=0.25, random_state=1):
        self_0 = set_field_wrapper(self, 'scoring', scoring)
        _var90 = clone(estimator)
        self_1 = set_field_wrapper(self_0, 'estimator', _var90)
        self_2 = set_field_wrapper(self_1, 'k_features', k_features)
        self_3 = set_field_wrapper(self_2, 'test_size', test_size)
        self_4 = set_field_wrapper(self_3, 'random_state', random_state)

    def fit(self_5, X_0, Y):
        _var95 = self_5.test_size
        _var96 = self_5.random_state
        (_var91, _var92, _var93, _var94) = train_test_split(X_0, Y, test_size=_var95, random_state=_var96)
        x_train = _var91
        y_train_0 = _var92
        x_test = _var93
        y_test_0 = _var94
        _var97 = np.shape(x_train)
        _var98 = 1
        dim = _var97[_var98]
        _var99 = range(dim)
        _var100 = tuple(_var99)
        self_6 = set_field_wrapper(self_5, 'indices', _var100)
        _var101 = self_6.indices
        _var102 = [_var101]
        self_7 = set_field_wrapper(self_6, 'subSets_', _var102)
        _var103 = self_7.x_train
        _var104 = self_7.indices
        score = self_7.calScore(_var103, y_train_0, x_test, y_test_0, _var104)
        _var105 = [score]
        self_8 = set_field_wrapper(self_7, 'scores_', _var105)
        _var106 = self_8.k_features
        _var107 = (dim > _var106)
        while _var107:
            scores = []
            subSet = []
            _var108 = self_8.indices
            _var109 = 1
            _var110 = (dim - _var109)
            _var111 = combinations(_var108, _var110)
            for p in _var111:
                _var112 = self_8.x_train
                score_0 = self_8.calScore(_var112, y_train_0, x_test, y_test_0, p)
                scores.append(score_0)
                subSet.append(p)
            score_1 = __phi__(score_0, score)
            best = np.argmax(score_1)
            _var113 = subSet[best]
            self_9 = set_field_wrapper(self_8, 'indices', _var113)
            _var114 = self_9.subSets_
            _var115 = self_9.indices
            _var114.append(_var115)
            _var116 = 1
            dim_0 = (dim - _var116)
            _var117 = self_9.scores_
            _var118 = scores[best]
            _var117.append(_var118)
        self_10 = __phi__(self_9, self_8)
        dim_1 = __phi__(dim_0, dim)
        score_2 = __phi__(score_1, score)
        _var119 = self_10.scores_
        _var120 = (- 1)
        _var121 = _var119[_var120]
        self_11 = set_field_wrapper(self_10, 'k_score', _var121)
        return self_11

    def calScore(self_12, x_train_0, y_train_1, x_test_0, y_test_1, indices):
        _var122 = self_12.estimator
        _var122_0 = _var122.fit(x_train_0, y_train_1)
        _var123 = self_12.estimator
        y_pred = _var123.predict(x_test_0)
        score_3 = self_12.scoring(y_test_1, y_pred)
        return score_3

class SBS():

    def __init__(self_13, estimator_0, k_features_0, scoring_0=accuracy_score, test_size_0=0.25, random_state_0=1):
        self_14 = set_field_wrapper(self_13, 'scoring', scoring_0)
        _var124 = clone(estimator_0)
        self_15 = set_field_wrapper(self_14, 'estimator', _var124)
        self_16 = set_field_wrapper(self_15, 'k_features', k_features_0)
        self_17 = set_field_wrapper(self_16, 'test_size', test_size_0)
        self_18 = set_field_wrapper(self_17, 'random_state', random_state_0)

    def fit(self_19, X_1, y_0):
        _var129 = self_19.test_size
        _var130 = self_19.random_state
        (_var125, _var126, _var127, _var128) = train_test_split(X_1, y_0, test_size=_var129, random_state=_var130)
        X_train_0 = _var125
        X_test_0 = _var126
        y_train_2 = _var127
        y_test_2 = _var128
        _var131 = X_train_0.shape
        _var132 = 1
        dim_2 = _var131[_var132]
        _var133 = range(dim_2)
        _var134 = tuple(_var133)
        self_20 = set_field_wrapper(self_19, 'indices_', _var134)
        _var135 = self_20.indices_
        _var136 = [_var135]
        self_21 = set_field_wrapper(self_20, 'subsets_', _var136)
        _var137 = self_21.indices_
        score_4 = self_21._calc_score(X_train_0, y_train_2, X_test_0, y_test_2, _var137)
        _var138 = [score_4]
        self_22 = set_field_wrapper(self_21, 'scores_', _var138)
        _var139 = self_22.k_features
        _var140 = (dim_2 > _var139)
        while _var140:
            scores_0 = []
            subsets = []
            _var141 = self_22.indices_
            _var142 = 1
            _var143 = (dim_2 - _var142)
            _var144 = combinations(_var141, r=_var143)
            for p_0 in _var144:
                score_5 = self_22._calc_score(X_train_0, y_train_2, X_test_0, y_test_2, p_0)
                scores_0.append(score_5)
                subsets.append(p_0)
                best_0 = np.argmax(scores_0)
                _var145 = subsets[best_0]
                self_23 = set_field_wrapper(self_22, 'indices_', _var145)
                _var146 = self_23.subsets_
                _var147 = self_23.indices_
                _var146.append(_var147)
            self_24 = __phi__(self_23, self_22)
            score_6 = __phi__(score_5, score_4)
            _var148 = 1
            dim_3 = (dim_2 - _var148)
            _var149 = self_24.scores_
            _var150 = scores_0[best_0]
            _var149.append(_var150)
        self_25 = __phi__(self_24, self_22)
        dim_4 = __phi__(dim_3, dim_2)
        score_7 = __phi__(score_6, score_4)
        _var151 = self_25.scores_
        _var152 = (- 1)
        _var153 = _var151[_var152]
        self_26 = set_field_wrapper(self_25, 'k_score_', _var153)
        return self_26

    def _calc_score(self_27, x_train_1, y_train_3, x_test_1, y_test_3, indices_0):
        _var154 = self_27.estimator
        _var154_0 = _var154.fit(x_train_1, y_train_3)
        _var155 = self_27.estimator
        y_pred_0 = _var155.predict(x_test_1)
        score_8 = self_27.scoring(y_test_3, y_pred_0)
        return score_8
from sklearn.neighbors import KNeighborsClassifier
_var156 = 2
knn = KNeighborsClassifier(n_neighbors=_var156)
_var157 = 1
sbs = SBS(knn, k_features_0=_var157)
sbs_0 = sbs.fit(X_train_std, y_train)
import matplotlib.pyplot as plt
fig_0 = plt.figure()
k_fea = [len(k) for k in sbs.subsets_]
_var158 = sbs_0.scores_
_var159 = 'o'
plt.plot(k_fea, _var158, marker=_var159)
plt.show()
