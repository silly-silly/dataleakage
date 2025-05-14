

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
from sklearn.datasets import load_digits
digits = load_digits()
_var0 = digits.keys()
list(_var0)
_var1 = digits.images
_var1.shape
_var2 = digits.images
_var3 = 0
_var4 = _var2[_var3]
print(_var4)
import matplotlib.pyplot as plt
_var5 = get_ipython()
_var6 = 'matplotlib'
_var7 = 'notebook'
_var5.run_line_magic(_var6, _var7)
_var8 = digits.images
_var9 = 0
_var10 = _var8[_var9]
_var11 = plt.cm
_var12 = _var11.Greys
plt.matshow(_var10, cmap=_var12)
_var13 = digits.data
_var13.shape
_var14 = digits.target
_var14.shape
digits.target
from sklearn.cross_validation import train_test_split
_var19 = digits.data
_var20 = digits.target
(_var15, _var16, _var17, _var18) = train_test_split(_var19, _var20)
X_train = _var15
X_test = _var16
y_train = _var17
y_test = _var18
from sklearn.cross_validation import train_test_split
import pandas as pd
_var21 = '/Users/shermanash/ds/metis/nyc16_ds6/04-mcnulty1/04-svms/cleveland_full_1.csv'
df = pd.read_csv(_var21)
cols_we_like = [col for col in df.columns if (col not in [['id', 'location', 'num']])]
X = df[cols_we_like]
y = df.num
(_var22, _var23, _var24, _var25) = train_test_split(X, y)
X_train_0 = _var22
X_test_0 = _var23
y_train_0 = _var24
y_test_0 = _var25
from sklearn.svm import LinearSVC
_var26 = 0.1
svm = LinearSVC(C=_var26)
svm_0 = svm.fit(X_train_0, y_train_0)
_var27 = svm_0.predict(X_train_0)
print(_var27)
print(y_train_0)
svm_0.score(X_train_0, y_train_0)
svm_0.score(X_test_0, y_test_0)
from sklearn.ensemble import RandomForestClassifier
_var28 = 50
rf = RandomForestClassifier(n_estimators=_var28)
rf_0 = rf.fit(X_train_0, y_train_0)
rf_0.score(X_test_0, y_test_0)
import numpy as np
import pylab as pl
from matplotlib.colors import ListedColormap
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.lda import LDA
from sklearn.qda import QDA
h = 0.02
names = ['Nearest Neighbors', 'Linear SVM', 'RBF SVM', 'Decision Tree', 'Random Forest', 'AdaBoost', 'Naive Bayes', 'LDA', 'QDA']
_var29 = 3
_var30 = KNeighborsClassifier(_var29)
_var31 = 'linear'
_var32 = 0.025
_var33 = SVC(kernel=_var31, C=_var32)
_var34 = 2
_var35 = 1
_var36 = SVC(gamma=_var34, C=_var35)
_var37 = 5
_var38 = DecisionTreeClassifier(max_depth=_var37)
_var39 = 5
_var40 = 10
_var41 = 1
_var42 = RandomForestClassifier(max_depth=_var39, n_estimators=_var40, max_features=_var41)
_var43 = AdaBoostClassifier()
_var44 = GaussianNB()
_var45 = LDA()
_var46 = QDA()
classifiers = [_var30, _var33, _var36, _var38, _var42, _var43, _var44, _var45, _var46]
_var49 = 2
_var50 = 0
_var51 = 2
_var52 = 1
_var53 = 1
(_var47, _var48) = make_classification(n_features=_var49, n_redundant=_var50, n_informative=_var51, random_state=_var52, n_clusters_per_class=_var53)
X_0 = _var47
y_0 = _var48
_var54 = np.random
_var55 = 2
rng = _var54.RandomState(_var55)
_var56 = 2
_var57 = X_0.shape
_var58 = rng.uniform(size=_var57)
_var59 = (_var56 * _var58)
X_1 = (X_0 + _var59)
linearly_separable = (X_1, y_0)
_var60 = 0.3
_var61 = 0
_var62 = make_moons(noise=_var60, random_state=_var61)
_var63 = 0.2
_var64 = 0.5
_var65 = 1
_var66 = make_circles(noise=_var63, factor=_var64, random_state=_var65)
datasets = [_var62, _var66, linearly_separable]
_var67 = 27
_var68 = 9
_var69 = (_var67, _var68)
figure = pl.figure(figsize=_var69)
i = 1
for ds in datasets:
    _var72 = 0
    X_2 = ds[_var72]
    _var73 = 1
    y_1 = ds[_var73]
    _var74 = StandardScaler()
    X_3 = _var74.fit_transform(X_2)
    _var79 = 0.4
    (_var75, _var76, _var77, _var78) = train_test_split(X_3, y_1, test_size=_var79)
    X_train_1 = _var75
    X_test_1 = _var76
    y_train_1 = _var77
    y_test_1 = _var78
    _var80 = 0
    _var81 = X_3[:, _var80]
    _var82 = _var81.min()
    _var83 = 0.5
    _var84 = (_var82 - _var83)
    _var85 = 0
    _var86 = X_3[:, _var85]
    _var87 = _var86.max()
    _var88 = 0.5
    _var89 = (_var87 + _var88)
    x_min = _var84
    x_max = _var89
    _var90 = 1
    _var91 = X_3[:, _var90]
    _var92 = _var91.min()
    _var93 = 0.5
    _var94 = (_var92 - _var93)
    _var95 = 1
    _var96 = X_3[:, _var95]
    _var97 = _var96.max()
    _var98 = 0.5
    _var99 = (_var97 + _var98)
    y_min = _var94
    y_max = _var99
    _var102 = np.arange(x_min, x_max, h)
    _var103 = np.arange(y_min, y_max, h)
    (_var100, _var101) = np.meshgrid(_var102, _var103)
    xx = _var100
    yy = _var101
    _var104 = pl.cm
    cm = _var104.RdBu
    _var105 = ['#FF0000', '#0000FF']
    cm_bright = ListedColormap(_var105)
    _var106 = len(datasets)
    _var107 = len(classifiers)
    _var108 = 1
    _var109 = (_var107 + _var108)
    ax = pl.subplot(_var106, _var109, i)
    _var110 = 0
    _var111 = X_train_1[:, _var110]
    _var112 = 1
    _var113 = X_train_1[:, _var112]
    ax.scatter(_var111, _var113, c=y_train_1, cmap=cm_bright)
    _var114 = 0
    _var115 = X_test_1[:, _var114]
    _var116 = 1
    _var117 = X_test_1[:, _var116]
    _var118 = 0.6
    ax.scatter(_var115, _var117, c=y_test_1, cmap=cm_bright, alpha=_var118)
    _var119 = xx.min()
    _var120 = xx.max()
    ax.set_xlim(_var119, _var120)
    _var121 = yy.min()
    _var122 = yy.max()
    ax.set_ylim(_var121, _var122)
    _var123 = ()
    ax.set_xticks(_var123)
    _var124 = ()
    ax.set_yticks(_var124)
    _var125 = 1
    i_0 = (i + _var125)
    _var126 = zip(names, classifiers)
    for _var127 in _var126:
        _var130 = 0
        name = _var127[_var130]
        _var131 = 1
        clf = _var127[_var131]
        _var132 = len(datasets)
        _var133 = len(classifiers)
        _var134 = 1
        _var135 = (_var133 + _var134)
        ax_0 = pl.subplot(_var132, _var135, i_0)
        clf_0 = clf.fit(X_train_1, y_train_1)
        score = clf_0.score(X_test_1, y_test_1)
        _var136 = 'decision_function'
        _var137 = hasattr(clf_0, _var136)
        if _var137:
            _var138 = np.c_
            _var139 = xx.ravel()
            _var140 = yy.ravel()
            _var141 = (_var139, _var140)
            _var142 = _var138[_var141]
            Z = clf_0.decision_function(_var142)
        else:
            _var143 = np.c_
            _var144 = xx.ravel()
            _var145 = yy.ravel()
            _var146 = (_var144, _var145)
            _var147 = _var143[_var146]
            _var148 = clf_0.predict_proba(_var147)
            _var149 = 1
            Z_0 = _var148[:, _var149]
        Z_1 = __phi__(Z, Z_0)
        _var150 = xx.shape
        Z_2 = Z_1.reshape(_var150)
        _var151 = 0.8
        ax_0.contourf(xx, yy, Z_2, cmap=cm, alpha=_var151)
        _var152 = 0
        _var153 = X_train_1[:, _var152]
        _var154 = 1
        _var155 = X_train_1[:, _var154]
        ax_0.scatter(_var153, _var155, c=y_train_1, cmap=cm_bright)
        _var156 = 0
        _var157 = X_test_1[:, _var156]
        _var158 = 1
        _var159 = X_test_1[:, _var158]
        _var160 = 0.6
        ax_0.scatter(_var157, _var159, c=y_test_1, cmap=cm_bright, alpha=_var160)
        _var161 = xx.min()
        _var162 = xx.max()
        ax_0.set_xlim(_var161, _var162)
        _var163 = yy.min()
        _var164 = yy.max()
        ax_0.set_ylim(_var163, _var164)
        _var165 = ()
        ax_0.set_xticks(_var165)
        _var166 = ()
        ax_0.set_yticks(_var166)
        ax_0.set_title(name)
        _var167 = xx.max()
        _var168 = 0.3
        _var169 = (_var167 - _var168)
        _var170 = yy.min()
        _var171 = 0.3
        _var172 = (_var170 + _var171)
        _var173 = '%.2f'
        _var174 = (_var173 % score)
        _var175 = '0'
        _var176 = _var174.lstrip(_var175)
        _var177 = 15
        _var178 = 'right'
        ax_0.text(_var169, _var172, _var176, size=_var177, horizontalalignment=_var178)
        _var179 = 1
        i_1 = (i_0 + _var179)
    i_2 = __phi__(i_1, i_0)
    ax_1 = __phi__(ax_0, ax)
y_2 = __phi__(y_1, y_0)
i_3 = __phi__(i_2, i)
X_train_2 = __phi__(X_train_1, X_train_0)
y_train_2 = __phi__(y_train_1, y_train_0)
X_test_2 = __phi__(X_test_1, X_test_0)
y_test_2 = __phi__(y_test_1, y_test_0)
X_4 = __phi__(X_3, X_1)
_var180 = 0.02
_var181 = 0.98
figure.subplots_adjust(left=_var180, right=_var181)
