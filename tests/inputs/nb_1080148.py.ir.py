

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
_var0 = get_ipython()
_var1 = 'matplotlib'
_var2 = 'inline'
_var0.run_line_magic(_var1, _var2)
import numpy as np
import scipy as sp
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import pandas as pd
_var3 = 'display.width'
_var4 = 500
pd.set_option(_var3, _var4)
_var5 = 'display.max_columns'
_var6 = 100
pd.set_option(_var5, _var6)
_var7 = 'display.notebook_repr_html'
_var8 = True
pd.set_option(_var7, _var8)
import seaborn as sns
_var9 = 'whitegrid'
sns.set_style(_var9)
_var10 = 'poster'
sns.set_context(_var10)
_var11 = sns.color_palette()
_var12 = 0
c0 = _var11[_var12]
_var13 = sns.color_palette()
_var14 = 1
c1 = _var13[_var14]
_var15 = sns.color_palette()
_var16 = 2
c2 = _var15[_var16]
from matplotlib.colors import ListedColormap
_var17 = ['#FFAAAA', '#AAFFAA', '#AAAAFF']
cmap_light = ListedColormap(_var17)
_var18 = ['#FF0000', '#00FF00', '#0000FF']
cmap_bold = ListedColormap(_var18)
_var19 = plt.cm
cm = _var19.RdBu
_var20 = ['#FF0000', '#0000FF']
cm_bright = ListedColormap(_var20)

def points_plot(ax, Xtr, Xte, ytr, yte, clf, mesh=True, colorscale=cmap_light, cdiscrete=cmap_bold, alpha=0.1, psize=10, zfunc=False, predicted=False):
    h = 0.02
    _var21 = (Xtr, Xte)
    X = np.concatenate(_var21)
    _var22 = 0
    _var23 = X[:, _var22]
    _var24 = _var23.min()
    _var25 = 0.5
    _var26 = (_var24 - _var25)
    _var27 = 0
    _var28 = X[:, _var27]
    _var29 = _var28.max()
    _var30 = 0.5
    _var31 = (_var29 + _var30)
    x_min = _var26
    x_max = _var31
    _var32 = 1
    _var33 = X[:, _var32]
    _var34 = _var33.min()
    _var35 = 0.5
    _var36 = (_var34 - _var35)
    _var37 = 1
    _var38 = X[:, _var37]
    _var39 = _var38.max()
    _var40 = 0.5
    _var41 = (_var39 + _var40)
    y_min = _var36
    y_max = _var41
    _var44 = 100
    _var45 = np.linspace(x_min, x_max, _var44)
    _var46 = 100
    _var47 = np.linspace(y_min, y_max, _var46)
    (_var42, _var43) = np.meshgrid(_var45, _var47)
    xx = _var42
    yy = _var43
    if zfunc:
        _var48 = np.c_
        _var49 = xx.ravel()
        _var50 = yy.ravel()
        _var51 = (_var49, _var50)
        _var52 = _var48[_var51]
        _var53 = clf.predict_proba(_var52)
        _var54 = 0
        p0 = _var53[:, _var54]
        _var55 = np.c_
        _var56 = xx.ravel()
        _var57 = yy.ravel()
        _var58 = (_var56, _var57)
        _var59 = _var55[_var58]
        _var60 = clf.predict_proba(_var59)
        _var61 = 1
        p1 = _var60[:, _var61]
        Z = zfunc(p0, p1)
    else:
        _var62 = np.c_
        _var63 = xx.ravel()
        _var64 = yy.ravel()
        _var65 = (_var63, _var64)
        _var66 = _var62[_var65]
        Z_0 = clf.predict(_var66)
    Z_1 = __phi__(Z, Z_0)
    _var67 = xx.shape
    ZZ = Z_1.reshape(_var67)
    if mesh:
        _var68 = global_wrapper(cmap_light)
        plt.pcolormesh(xx, yy, ZZ, cmap=_var68, alpha=alpha, axes=ax)
    if predicted:
        showtr = clf.predict(Xtr)
        showte = clf.predict(Xte)
    else:
        showtr_0 = ytr
        showte_0 = yte
    showtr_1 = __phi__(showtr, showtr_0)
    showte_1 = __phi__(showte, showte_0)
    _var69 = 0
    _var70 = Xtr[:, _var69]
    _var71 = 1
    _var72 = Xtr[:, _var71]
    _var73 = 1
    _var74 = (showtr_1 - _var73)
    _var75 = global_wrapper(cmap_bold)
    _var76 = 'k'
    ax.scatter(_var70, _var72, c=_var74, cmap=_var75, s=psize, alpha=alpha, edgecolor=_var76)
    _var77 = 0
    _var78 = Xte[:, _var77]
    _var79 = 1
    _var80 = Xte[:, _var79]
    _var81 = 1
    _var82 = (showte_1 - _var81)
    _var83 = global_wrapper(cmap_bold)
    _var84 = 's'
    _var85 = 10
    _var86 = (psize + _var85)
    ax.scatter(_var78, _var80, c=_var82, cmap=_var83, alpha=alpha, marker=_var84, s=_var86)
    _var87 = xx.min()
    _var88 = xx.max()
    ax.set_xlim(_var87, _var88)
    _var89 = yy.min()
    _var90 = yy.max()
    ax.set_ylim(_var89, _var90)
    return (ax, xx, yy)

def points_plot_prob(ax_0, Xtr_0, Xte_0, ytr_0, yte_0, clf_0, colorscale_0=cmap_light, cdiscrete_0=cmap_bold, ccolor=cm, psize_0=10, alpha_0=0.1):
    _var94 = False
    _var95 = True
    (_var91, _var92, _var93) = points_plot(ax_0, Xtr_0, Xte_0, ytr_0, yte_0, clf_0, mesh=_var94, colorscale=colorscale_0, cdiscrete=cdiscrete_0, psize=psize_0, alpha=alpha_0, predicted=_var95)
    ax_1 = _var91
    xx_0 = _var92
    yy_0 = _var93
    _var96 = np.c_
    _var97 = xx_0.ravel()
    _var98 = yy_0.ravel()
    _var99 = (_var97, _var98)
    _var100 = _var96[_var99]
    _var101 = clf_0.predict_proba(_var100)
    _var102 = 1
    Z_2 = _var101[:, _var102]
    _var103 = xx_0.shape
    Z_3 = Z_2.reshape(_var103)
    _var104 = 0.2
    plt.contourf(xx_0, yy_0, Z_3, cmap=ccolor, alpha=_var104, axes=ax_1)
    _var105 = 0.6
    cs2 = plt.contour(xx_0, yy_0, Z_3, cmap=ccolor, alpha=_var105, axes=ax_1)
    _var106 = '%2.1f'
    _var107 = 'k'
    _var108 = 14
    plt.clabel(cs2, fmt=_var106, colors=_var107, fontsize=_var108, axes=ax_1)
    return ax_1
_var109 = 'data/01_heights_weights_genders.csv'
dflog = pd.read_csv(_var109)
dflog.head()
import matplotlib.pyplot as plt
_var110 = dflog.Gender
_var111 = 'Male'
_var112 = (_var110 == _var111)
male = dflog[_var112]
_var113 = dflog.Gender
_var114 = 'Female'
_var115 = (_var113 == _var114)
female = dflog[_var115]
fig = plt.figure()
_var116 = 111
ax1 = fig.add_subplot(_var116)
_var117 = male.Weight
_var118 = male.Height
_var119 = 'b'
_var120 = 'Male'
ax1.scatter(_var117, _var118, c=_var119, label=_var120)
_var121 = female.Weight
_var122 = female.Height
_var123 = 'r'
_var124 = 'Female'
ax1.scatter(_var121, _var122, c=_var123, label=_var124)
_var125 = 'upper left'
plt.legend(loc=_var125)
_var126 = 'Male & Female: Weight vs Height'
plt.title(_var126)
_var127 = 'Weight'
plt.xlabel(_var127)
_var128 = 'Height'
plt.ylabel(_var128)
plt.show()
from sklearn.cross_validation import KFold
from sklearn.metrics import accuracy_score

def cv_score(clf_1, x, y, score_func=accuracy_score):
    result = 0
    nfold = 5
    _var129 = y.size
    _var130 = KFold(_var129, nfold)
    for _var131 in _var130:
        _var134 = 0
        train = _var131[_var134]
        _var135 = 1
        test = _var131[_var135]
        _var136 = x[train]
        _var137 = y[train]
        clf_2 = clf_1.fit(_var136, _var137)
        _var138 = x[test]
        _var139 = clf_2.predict(_var138)
        _var140 = y[test]
        _var141 = score_func(_var139, _var140)
        result_0 = (result + _var141)
    result_1 = __phi__(result_0, result)
    clf_3 = __phi__(clf_2, clf_1)
    _var142 = (result_1 / nfold)
    return _var142
from sklearn.cross_validation import train_test_split
_var147 = ['Height', 'Weight']
_var148 = dflog[_var147]
_var149 = _var148.values
_var150 = dflog.Gender
_var151 = 'Male'
_var152 = (_var150 == _var151)
_var153 = _var152.values
_var154 = 5
(_var143, _var144, _var145, _var146) = train_test_split(_var149, _var153, random_state=_var154)
Xlr = _var143
Xtestlr = _var144
ylr = _var145
ytestlr = _var146
_var155 = len(Xlr)
_var156 = len(Xtestlr)
_var157 = ['Height', 'Weight']
_var158 = dflog[_var157]
_var159 = _var158.values
_var160 = len(_var159)
_var161 = dflog.Gender
_var162 = 'Male'
_var163 = (_var161 == _var162)
_var164 = _var163.values
_var165 = len(_var164)
(_var155, _var156, _var160, _var165)
from sklearn.linear_model import LogisticRegression
clf_4 = LogisticRegression()
clf_5 = clf_4.fit(Xlr, ylr)
_var166 = clf_5.predict(Xtestlr)
_var167 = accuracy_score(_var166, ytestlr)
print(_var167)
clf_6 = LogisticRegression()
score = cv_score(clf_6, Xlr, ylr)
print(score)
Cs = [0.001, 0.1, 1, 10, 100]
scores_per_C = {}
for c in Cs:
    clf_7 = LogisticRegression(C=c)
    _var168 = cv_score(clf_7, Xlr, ylr)
    _var169 = {c: _var168}
    scores_per_C.update(_var169)
clf_8 = __phi__(clf_7, clf_6)
_var170 = scores_per_C.values()
v = list(_var170)
_var171 = scores_per_C.keys()
k = list(_var171)
_var172 = max(v)
_var173 = v.index(_var172)
C = k[_var173]
scores_per_C
clf_9 = LogisticRegression(C=C)
clf_10 = clf_9.fit(Xlr, ylr)
_var174 = clf_10.predict(Xtestlr)
_var175 = accuracy_score(_var174, ytestlr)
print(_var175)
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
tuned_parameters = {'C': Cs}
_var176 = 1
_var177 = SVC(C=_var176)
_var178 = 5
clf_11 = GridSearchCV(_var177, tuned_parameters, cv=_var178)
clf_12 = clf_11.fit(Xlr, ylr)
_var179 = 'Best parameters set found on development set:'
print(_var179)
_var180 = clf_12.best_params_
print(_var180)

def cv_optimize(clf_13, parameters, Xtrain, ytrain, n_folds=5):
    gs = GridSearchCV(clf_13, param_grid=parameters, cv=n_folds)
    gs_0 = gs.fit(Xtrain, ytrain)
    _var181 = 'BEST PARAMS'
    _var182 = gs_0.best_params_
    _var183 = (_var181, _var182)
    print(_var183)
    best = gs_0.best_estimator_
    return best
from sklearn.cross_validation import train_test_split

def do_classify(clf_14, parameters_0, indf, featurenames, targetname, target1val, standardize=False, train_size=0.8):
    subdf = indf[featurenames]
    if standardize:
        _var184 = subdf.mean()
        _var185 = (subdf - _var184)
        _var186 = subdf.std()
        subdfstd = (_var185 / _var186)
    else:
        subdfstd_0 = subdf
    subdfstd_1 = __phi__(subdfstd, subdfstd_0)
    X_0 = subdfstd_1.values
    _var187 = indf[targetname]
    _var188 = _var187.values
    _var189 = (_var188 == target1val)
    _var190 = 1
    y_0 = (_var189 * _var190)
    (_var191, _var192, _var193, _var194) = train_test_split(X_0, y_0, train_size=train_size)
    Xtrain_0 = _var191
    Xtest = _var192
    ytrain_0 = _var193
    ytest = _var194
    clf_15 = cv_optimize(clf_14, parameters_0, Xtrain_0, ytrain_0)
    clf_16 = clf_15.fit(Xtrain_0, ytrain_0)
    clf_17 = clf_16
    training_accuracy = clf_17.score(Xtrain_0, ytrain_0)
    test_accuracy = clf_17.score(Xtest, ytest)
    _var195 = 'Accuracy on training data: %0.2f'
    _var196 = (_var195 % training_accuracy)
    print(_var196)
    _var197 = 'Accuracy on test data:     %0.2f'
    _var198 = (_var197 % test_accuracy)
    print(_var198)
    return (clf_17, Xtrain_0, ytrain_0, Xtest, ytest)

def _func0(z):
    _var199 = 1.0
    _var200 = 1
    _var201 = (- z)
    _var202 = np.exp(_var201)
    _var203 = (_var200 + _var202)
    _var204 = (_var199 / _var203)
    return _var204
h_0 = _func0
_var205 = (- 5)
_var206 = 5
_var207 = 0.1
zs = np.arange(_var205, _var206, _var207)
_var208 = h_0(zs)
_var209 = 0.5
plt.plot(zs, _var208, alpha=_var209)
dflog.head()
_var215 = LogisticRegression()
_var216 = [0.01, 0.1, 1, 10, 100]
_var217 = {'C': _var216}
_var218 = ['Weight', 'Height']
_var219 = 'Gender'
_var220 = 'Male'
(_var210, _var211, _var212, _var213, _var214) = do_classify(_var215, _var217, dflog, _var218, _var219, _var220)
clf_l = _var210
Xtrain_l = _var211
ytrain_l = _var212
Xtest_l = _var213
ytest_l = _var214
plt.figure()
ax_2 = plt.gca()
_var221 = 0.2
points_plot(ax_2, Xtrain_l, Xtest_l, ytrain_l, ytest_l, clf_l, alpha=_var221)
clf_l.predict_proba(Xtest_l)
plt.figure()
ax_3 = plt.gca()
_var222 = 20
_var223 = 0.1
points_plot_prob(ax_3, Xtrain_l, Xtest_l, ytrain_l, ytest_l, clf_l, psize_0=_var222, alpha_0=_var223)
