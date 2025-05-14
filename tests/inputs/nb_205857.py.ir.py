

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
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
import urllib.request, urllib.parse, urllib.error
import pandas
_var0 = get_ipython()
_var1 = 'matplotlib'
_var2 = 'inline'
_var0.run_line_magic(_var1, _var2)
url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data'
_var3 = global_wrapper(urllib)
_var4 = _var3.request
raw_data = _var4.urlopen(url)
_var5 = ','
dataset = np.loadtxt(raw_data, delimiter=_var5)
target = [int(x[0]) for x in dataset]
dataset_0 = [x[1:] for x in dataset]
X = dataset_0
Y = np.int_(target)
X_0 = preprocessing.scale(dataset_0)
_var10 = 0.3
_var11 = 0.7
(_var6, _var7, _var8, _var9) = train_test_split(X_0, Y, test_size=_var10, train_size=_var11)
traintX = _var6
testX = _var7
traintY = _var8
testY = _var9
_var12 = len(traintX)
_var13 = len(traintY)
_var14 = (_var12, _var13)
print(_var14)
_var15 = len(testX)
_var16 = len(testY)
_var17 = (_var15, _var16)
print(_var17)
from collections import OrderedDict
_var18 = True
_var19 = True
_var20 = 'sqrt'
RandomSqrt = RandomForestClassifier(warm_start=_var18, oob_score=_var19, max_features=_var20)
_var21 = True
_var22 = True
_var23 = 'log2'
Randomlog2 = RandomForestClassifier(warm_start=_var21, oob_score=_var22, max_features=_var23)
_var24 = True
_var25 = True
_var26 = None
RandomNone = RandomForestClassifier(warm_start=_var24, oob_score=_var25, max_features=_var26)
_var27 = "RandomForestClassifier, max_features='sqrt'"
_var28 = (_var27, RandomSqrt)
_var29 = "RandomForestClassifier, max_features='log2'"
_var30 = (_var29, Randomlog2)
_var31 = 'RandomForestClassifier, max_features=None'
_var32 = (_var31, RandomNone)
arreglo = [_var28, _var30, _var32]
_var33 = ((label, []) for (label, _) in arreglo)
tasa_error = OrderedDict(_var33)
_var34 = ((label, []) for (label, _) in arreglo)
estimacion = OrderedDict(_var34)
min_estimators = 1
max_estimators = 200
for _var35 in arreglo:
    _var38 = 0
    label = _var35[_var38]
    _var39 = 1
    rf = _var35[_var39]
    _var40 = 1
    _var41 = (max_estimators + _var40)
    _var42 = range(min_estimators, _var41)
    for i in _var42:
        rf.set_params(n_estimators=i)
        rf_0 = rf.fit(traintX, traintY)
        _var43 = 1
        _var44 = rf_0.oob_score
        error_ob = (_var43 - _var44)
        _var45 = tasa_error[label]
        _var46 = (i, error_ob)
        _var45.append(_var46)
        prediction = rf_0.predict(testX)
        valor = rf_0.score(testX, testY)
        _var47 = estimacion[label]
        _var48 = (i, valor)
        _var47.append(_var48)
    rf_1 = __phi__(rf_0, rf)
_var49 = tasa_error.items()
_var50 = list(_var49)
for _var51 in _var50:
    _var54 = 0
    label_0 = _var51[_var54]
    _var55 = 1
    clf_err = _var51[_var55]
    _var58 = zip(*clf_err)
    (_var56, _var57) = list(_var58)
    xs = _var56
    ys = _var57
    plt.plot(xs, ys, label=label_0)
label_1 = __phi__(label_0, label)
plt.xlim(min_estimators, max_estimators)
_var59 = 'No estimadores'
plt.xlabel(_var59)
_var60 = 'tasa de error_oob'
plt.ylabel(_var60)
_var61 = 'upper right'
plt.legend(loc=_var61)
plt.show()
_var62 = estimacion.items()
_var63 = list(_var62)
for _var64 in _var63:
    _var67 = 0
    label_2 = _var64[_var67]
    _var68 = 1
    clf_err_0 = _var64[_var68]
    _var71 = zip(*clf_err_0)
    (_var69, _var70) = list(_var71)
    xs_0 = _var69
    ys_0 = _var70
    plt.plot(xs_0, ys_0, label=label_2)
ys_1 = __phi__(ys_0, ys)
xs_1 = __phi__(xs_0, xs)
label_3 = __phi__(label_2, label_1)
clf_err_1 = __phi__(clf_err_0, clf_err)
plt.xlim(min_estimators, max_estimators)
_var72 = 'No estimadores'
plt.xlabel(_var72)
_var73 = 'Presicion Media'
plt.ylabel(_var73)
_var74 = 'lower right'
plt.legend(loc=_var74)
_var75 = 0.9
_var76 = 1.01
plt.ylim(_var75, _var76)
_var77 = 0.0
_var78 = 140
plt.xlim(_var77, _var78)
plt.show()
print()
_var79 = estimacion.items()
_var80 = list(_var79)
for _var81 in _var80:
    _var84 = 0
    label_4 = _var81[_var84]
    _var85 = 1
    clf_err_2 = _var81[_var85]
    _var88 = zip(*clf_err_2)
    (_var86, _var87) = list(_var88)
    xs_2 = _var86
    ys_2 = _var87
    plt.plot(xs_2, ys_2, label=label_4)
ys_3 = __phi__(ys_2, ys_1)
xs_3 = __phi__(xs_2, xs_1)
label_5 = __phi__(label_4, label_3)
clf_err_3 = __phi__(clf_err_2, clf_err_1)
plt.xlim(min_estimators, max_estimators)
_var89 = 'No estimadores'
plt.xlabel(_var89)
_var90 = 'PrecisionMedia'
plt.ylabel(_var90)
_var91 = 'lower right'
plt.legend(loc=_var91)
_var92 = 0.9
_var93 = 1.01
plt.ylim(_var92, _var93)
_var94 = 0.0
_var95 = 10
plt.xlim(_var94, _var95)
plt.show()
n_components = 12
_var96 = 0
_var97 = 1
K = np.arange(_var96, n_components, _var97)
suma = 0.0
SUM = np.zeros(n_components)
pca = PCA(n_components)
pca_0 = pca.fit(traintX, traintY)
PVE = pca_0.explained_variance_ratio_
_var98 = 0
_var99 = range(_var98, n_components)
for i_0 in _var99:
    _var100 = PVE[i_0]
    suma_0 = (suma + _var100)
    SUM_0 = set_index_wrapper(SUM, i_0, suma_0)
SUM_1 = __phi__(SUM_0, SUM)
suma_1 = __phi__(suma_0, suma)
_var101 = 'black'
plt.scatter(K, SUM_1, color=_var101)
_var102 = 'red'
plt.plot(K, SUM_1, color=_var102)
_var103 = 'components'
plt.xlabel(_var103)
_var104 = 'explained_variance_ratio Acumulada'
plt.ylabel(_var104)
print(SUM_1)
S = pca_0.components_
_var105 = '1'
_var106 = 'C'
_var107 = (_var105 + _var106)
_var108 = '2'
_var109 = 'C'
_var110 = (_var108 + _var109)
_var111 = '3'
_var112 = 'C'
_var113 = (_var111 + _var112)
_var114 = '4'
_var115 = 'C'
_var116 = (_var114 + _var115)
_var117 = '5'
_var118 = 'C'
_var119 = (_var117 + _var118)
_var120 = '6'
_var121 = 'C'
_var122 = (_var120 + _var121)
_var123 = '7'
_var124 = 'C'
_var125 = (_var123 + _var124)
_var126 = '8'
_var127 = 'C'
_var128 = (_var126 + _var127)
_var129 = '9'
_var130 = 'C'
_var131 = (_var129 + _var130)
_var132 = '10'
_var133 = 'C'
_var134 = (_var132 + _var133)
_var135 = '11'
_var136 = 'C'
_var137 = (_var135 + _var136)
_var138 = '12'
_var139 = 'C'
_var140 = (_var138 + _var139)
teams_list1 = [_var107, _var110, _var113, _var116, _var119, _var122, _var125, _var128, _var131, _var134, _var137, _var140]
teams_list2 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
pandas.DataFrame(S, teams_list1, teams_list2)
n_clusters = 4
kmeans = KMeans(n_clusters)
kmeans_0 = kmeans.fit(traintX, traintY)
labels = kmeans_0.labels_
centroids = kmeans_0.cluster_centers_
_var141 = range(n_clusters)
for i_1 in _var141:
    _var142 = (labels == i_1)
    _var143 = np.where(_var142)
    ds = traintX[_var143]
    _var144 = 0
    _var145 = ds[:, _var144]
    _var146 = 1
    _var147 = ds[:, _var146]
    _var148 = 'o'
    plt.plot(_var145, _var147, _var148)
    _var149 = 0
    _var150 = (i_1, _var149)
    _var151 = centroids[_var150]
    _var152 = 1
    _var153 = (i_1, _var152)
    _var154 = centroids[_var153]
    _var155 = 'kx'
    lines = plt.plot(_var151, _var154, _var155)
    _var156 = 15.0
    plt.setp(lines, ms=_var156)
    _var157 = 2.0
    plt.setp(lines, mew=_var157)
_var158 = 'Case1'
plt.xlabel(_var158)
_var159 = 'Clase2'
plt.ylabel(_var159)
n_clusters_0 = 15
scoreR = np.zeros(n_clusters_0)
inerciaR = np.zeros(n_clusters_0)
_var160 = 0
_var161 = 1
cluster = np.arange(_var160, n_clusters_0, _var161)
_var162 = 1
_var163 = range(_var162, n_clusters_0)
for i_2 in _var163:
    kmeans_1 = KMeans(i_2)
    kmeans_2 = kmeans_1.fit(traintX, traintY)
    labels_0 = kmeans_2.labels_
    centroids_0 = kmeans_2.cluster_centers_
    _var164 = kmeans_2.score(traintX, traintY)
    scoreR_0 = set_index_wrapper(scoreR, i_2, _var164)
    _var165 = kmeans_2.inertia_
    inerciaR_0 = set_index_wrapper(inerciaR, i_2, _var165)
inerciaR_1 = __phi__(inerciaR_0, inerciaR)
scoreR_1 = __phi__(scoreR_0, scoreR)
labels_1 = __phi__(labels_0, labels)
centroids_1 = __phi__(centroids_0, centroids)
kmeans_3 = __phi__(kmeans_2, kmeans_0)
_var166 = (inerciaR_1, cluster)
print(_var166)
_var167 = 1
plt.xlim(_var167, n_clusters_0)
_var168 = 'No cluster'
plt.xlabel(_var168)
_var169 = 'score'
plt.ylabel(_var169)
plt.scatter(cluster, inerciaR_1)
plt.show()
n_components_0 = 7
pca_1 = PCA(n_components_0)
pca_2 = pca_1.fit(dataset_0)
Aux = pca_2.transform(dataset_0)
X_1 = preprocessing.scale(Aux)
_var174 = 0.3
_var175 = 0.7
(_var170, _var171, _var172, _var173) = train_test_split(X_1, Y, test_size=_var174, train_size=_var175)
traintX_0 = _var170
testX_0 = _var171
traintY_0 = _var172
testY_0 = _var173
from sklearn.metrics import classification_report
n_clusters_1 = 3
kmeans_4 = KMeans(n_clusters_1)
kmeans_5 = kmeans_4.fit(traintX_0, traintY_0)
prediction_0 = kmeans_5.predict(testX_0)
_var176 = classification_report(testY_0, prediction_0)
print(_var176)
_var177 = True
_var178 = 'log2'
_var179 = 4
rf1 = RandomForestClassifier(warm_start=_var177, max_features=_var178, n_estimators=_var179)
rf1_0 = rf1.fit(traintX_0, traintY_0)
prediction_1 = rf1_0.predict(testX_0)
_var180 = classification_report(testY_0, prediction_1)
print(_var180)
