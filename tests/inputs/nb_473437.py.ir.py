

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
from datetime import datetime
import shapefile
import csv
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
_var0 = get_ipython()
_var1 = 'matplotlib'
_var2 = 'inline'
_var0.run_line_magic(_var1, _var2)
from sklearn.cross_validation import train_test_split
from patsy import dmatrices
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import auc
import pickle
from sklearn.multiclass import OneVsRestClassifier
from sklearn import preprocessing as prp
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.linear_model import Perceptron
from sklearn import metrics
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import confusion_matrix
from pylab import *
import time
from datetime import datetime
_var3 = 'full_data.pkl'
_var4 = 'r'
f = open(_var3, _var4)
with open(_var3, _var4) as f:
    df_full = pickle.load(f)
_var5 = 'bikeids.csv'
df_bikeid = pd.read_csv(_var5)
_var6 = 'starttime_time'
_var7 = 'starttime'
_var8 = df_bikeid[_var7]

def _func0(x):
    _var9 = '%Y-%m-%d %H:%M:%S'
    _var10 = time.strptime(x, _var9)
    _var11 = _var10.tm_hour
    _var12 = 60
    _var13 = (_var11 * _var12)
    _var14 = '%Y-%m-%d %H:%M:%S'
    _var15 = time.strptime(x, _var14)
    _var16 = _var15.tm_min
    _var17 = (_var13 + _var16)
    return _var17
_var18 = _var8.map(_func0)
df_bikeid_0 = set_index_wrapper(df_bikeid, _var6, _var18)
_var19 = 'over_45'
_var20 = df_bikeid_0.tripduration

def _func1(x_0):
    _var21 = 2700
    _var22 = (x_0 > _var21)
    _var23 = 1
    _var24 = 0
    _var25 = (_var23 if _var22 else _var24)
    return _var25
_var26 = _var20.map(_func1)
df_bikeid_1 = set_index_wrapper(df_bikeid_0, _var19, _var26)
week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
_var27 = 'day_start'
_var28 = 'starttime'
_var29 = df_bikeid_1[_var28]

def _func2(x_1):
    _var30 = global_wrapper(week)
    _var31 = '%Y-%m-%d %H:%M:%S'
    _var32 = datetime.strptime(x_1, _var31)
    _var33 = _var32.weekday()
    _var34 = _var30[_var33]
    return _var34
_var35 = _var29.map(_func2)
df_bikeid_2 = set_index_wrapper(df_bikeid_1, _var27, _var35)
_var36 = 'starttime'
_var37 = 'starttime'
_var38 = df_bikeid_2[_var37]

def _func3(x_2):
    _var39 = '%Y-%m-%d %H:%M:%S'
    _var40 = datetime.strptime(x_2, _var39)
    return _var40
_var41 = _var38.map(_func3)
df_bikeid_3 = set_index_wrapper(df_bikeid_2, _var36, _var41)
_var42 = 'stoptime'
_var43 = 'stoptime'
_var44 = df_bikeid_3[_var43]

def _func4(x_3):
    _var45 = '%Y-%m-%d %H:%M:%S'
    _var46 = datetime.strptime(x_3, _var45)
    return _var46
_var47 = _var44.map(_func4)
df_bikeid_4 = set_index_wrapper(df_bikeid_3, _var42, _var47)
_var48 = 1
t = df_bikeid_4.shift(_var48)
_var49 = ['stoptime', 'end_station_id', 'n2', 'day_start', 'bikeid']
t_0 = t[_var49]
_var50 = [(x + '_') for x in t.columns]
t_1 = set_field_wrapper(t_0, 'columns', _var50)
_var51 = (t_1, df_bikeid_4)
_var52 = 1
df_bikes = pd.concat(_var51, _var52)
len(df_bikes)
_var53 = 'bikeid_'
_var54 = df_bikes[_var53]
_var55 = 'bikeid'
_var56 = df_bikeid_4[_var55]
_var57 = (_var54 == _var56)
df_bikes_0 = df_bikes[_var57]
_var58 = 'time_lapsed'

def _func5(row):
    _var59 = 'starttime'
    _var60 = row[_var59]
    _var61 = 'stoptime_'
    _var62 = row[_var61]
    _var63 = (_var60 - _var62)
    return _var63
_var64 = 1
_var65 = df_bikes_0.apply(_func5, axis=_var64)
df_bikes_1 = set_index_wrapper(df_bikes_0, _var58, _var65)
_var66 = 'same_station'

def _func6(row_0):
    _var67 = 'end_station_id_'
    _var68 = row_0[_var67]
    _var69 = 'start_station_id'
    _var70 = row_0[_var69]
    _var71 = (_var68 == _var70)
    _var72 = True
    _var73 = False
    _var74 = (_var72 if _var71 else _var73)
    return _var74
_var75 = 1
_var76 = df_bikes_1.apply(_func6, axis=_var75)
df_bikes_2 = set_index_wrapper(df_bikes_1, _var66, _var76)
_var77 = 'time_lapsed'
_var78 = df_bikes_2[_var77]
_var79 = 0
_var80 = (_var78 < _var79)
_var81 = df_bikes_2[_var80]
_var82 = _var81.bikeid
bad_bikes = _var82.unique()
_var83 = 'bikeid'
_var84 = df_bikes_2[_var83]
_var85 = _var84.isin(bad_bikes)
_var86 = (- _var85)
df_good_bikes = df_bikes_2[_var86]
_var87 = 'bikeid_data_good.pkl'
_var88 = 'wb'
f_0 = open(_var87, _var88)
with open(_var87, _var88) as f:
    pickle.dump(df_good_bikes, f_0)
_var89 = 'model_results_1.pkl'
_var90 = 'r'
f_1 = open(_var89, _var90)
with open(_var89, _var90) as f:
    r_1 = pickle.load(f_1)
_var91 = 'bikeid_data_good.pkl'
_var92 = 'r'
f_2 = open(_var91, _var92)
with open(_var91, _var92) as f:
    df = pickle.load(f_2)
_var93 = 'birth_year'
_var94 = df_full[_var93]
_var95 = 1915.0
_var96 = (_var94 > _var95)
_var97 = 'usertype'
_var98 = df_full[_var97]
_var99 = 'Subscriber'
_var100 = (_var98 == _var99)
_var101 = (_var96 & _var100)
_var102 = 'gender'
_var103 = df_full[_var102]
_var104 = 0
_var105 = (_var103 != _var104)
_var106 = (_var101 & _var105)
_var107 = 'tripduration'
_var108 = df_full[_var107]
_var109 = 3600
_var110 = (_var108 <= _var109)
_var111 = (_var106 & _var110)
df_sub = df_full[_var111]
_var112 = len(df_full)
print(_var112)
_var113 = len(df_sub)
print(_var113)
_var114 = 'cum_amt'
_var115 = 'cum_amt'
_var116 = df[_var115]

def _func7(f_3):
    _var117 = float(f_3)
    return _var117
_var118 = _var116.map(_func7)
df_0 = set_index_wrapper(df, _var114, _var118)

def assign_train_test(df_any, sub=True):
    'Assign y and X. \n    Dummify categorical variables (neighborhood, weekday, gender).\n    Standardize continuous variables.\n    Returns train and test subsets'
    _var119 = True
    _var120 = (sub == _var119)
    if _var120:
        _var121 = 'n2'
        y = df_any[_var121]
        cols_n = ['tripduration', 'birth_year', 'starttime_time', 'docks_y']
        cols_c = ['n1', 'gender', 'day_start']
    else:
        _var122 = False
        _var123 = (sub == _var122)
        if _var123:
            _var124 = 'n2'
            y_0 = df_any[_var124]
            cols_n_0 = ['tripduration', 'starttime_time', 'docks_y']
            cols_c_0 = ['n1', 'day_start', 'usertype']
        else:
            _var125 = 'bikes'
            _var126 = (sub == _var125)
            if _var126:
                _var127 = 'same_station'
                y_1 = df_any[_var127]
                cols_n_1 = ['cum_amt']
                cols_c_1 = ['n2_', 'day_start_']
            else:
                _var128 = 'n2'
                y_2 = df_any[_var128]
                cols_n_2 = ['birth_year', 'starttime_time']
                cols_c_2 = ['gender', 'day_start', 'n1']
            y_3 = __phi__(y_1, y_2)
            cols_c_3 = __phi__(cols_c_1, cols_c_2)
            cols_n_3 = __phi__(cols_n_1, cols_n_2)
        y_4 = __phi__(y_0, y_3)
        cols_c_4 = __phi__(cols_c_0, cols_c_3)
        cols_n_4 = __phi__(cols_n_0, cols_n_3)
    y_5 = __phi__(y, y_4)
    cols_c_5 = __phi__(cols_c, cols_c_4)
    cols_n_5 = __phi__(cols_n, cols_n_4)
    _var129 = (cols_n_5 + cols_c_5)
    X = df_any[_var129]
    _var134 = 0.25
    _var135 = 1
    (_var130, _var131, _var132, _var133) = train_test_split(X, y_5, test_size=_var134, random_state=_var135)
    X_train = _var130
    X_test = _var131
    y_train = _var132
    y_test = _var133
    scaler = StandardScaler()
    _var136 = X_train[cols_n_5]
    scaler_0 = scaler.fit(_var136)
    _var137 = X_train[cols_n_5]
    _var138 = scaler_0.transform(_var137)
    X_train_1 = pd.DataFrame(_var138, columns=cols_n_5)
    _var139 = X_test[cols_n_5]
    _var140 = scaler_0.transform(_var139)
    X_test_1 = pd.DataFrame(_var140, columns=cols_n_5)
    _var141 = X_train[cols_c_5]
    X_train_2 = pd.get_dummies(_var141, columns=cols_c_5)
    _var142 = X_test[cols_c_5]
    X_test_2 = pd.get_dummies(_var142, columns=cols_c_5)
    _var143 = True
    _var144 = X_train_1.reset_index(drop=_var143)
    _var145 = True
    _var146 = X_train_2.reset_index(drop=_var145)
    _var147 = (_var144, _var146)
    _var148 = 1
    X_train_0 = pd.concat(_var147, _var148)
    _var149 = True
    _var150 = X_test_1.reset_index(drop=_var149)
    _var151 = True
    _var152 = X_test_2.reset_index(drop=_var151)
    _var153 = (_var150, _var152)
    _var154 = 1
    X_test_0 = pd.concat(_var153, _var154)
    return (X_train_0, X_test_0, y_train, y_test)
_var159 = 'hey'
(_var155, _var156, _var157, _var158) = assign_train_test(df_sub, sub=_var159)
X_train_1 = _var155
X_test_1 = _var156
y_train_0 = _var157
y_test_0 = _var158

def conf_matrix(clf):
    _var160 = '_'
    _var161 = 80
    _var162 = (_var160 * _var161)
    print(_var162)
    _var163 = 'Training: '
    print(_var163)
    print(clf)
    t0 = time()
    _var164 = global_wrapper(X_train_1)
    _var165 = global_wrapper(y_train_0)
    clf_0 = clf.fit(_var164, _var165)
    _var166 = time()
    train_time = (_var166 - t0)
    _var167 = 'train time: %0.3fs'
    _var168 = (_var167 % train_time)
    print(_var168)
    t0_0 = time()
    _var169 = global_wrapper(X_test_1)
    pred = clf_0.predict(_var169)
    _var170 = time()
    test_time = (_var170 - t0_0)
    _var171 = 'test time:  %0.3fs'
    _var172 = (_var171 % test_time)
    print(_var172)
    _var173 = global_wrapper(y_test_0)
    _var174 = ['True', 'False']
    _var175 = confusion_matrix(_var173, pred, labels=_var174)
    return _var175

def benchmark(clf_1):
    _var176 = '_'
    _var177 = 80
    _var178 = (_var176 * _var177)
    print(_var178)
    _var179 = 'Training: '
    print(_var179)
    print(clf_1)
    t0_1 = time.time()
    _var180 = global_wrapper(X_train_1)
    _var181 = global_wrapper(y_train_0)
    clf_2 = clf_1.fit(_var180, _var181)
    _var182 = time.time()
    train_time_0 = (_var182 - t0_1)
    _var183 = 'train time: %0.3fs'
    _var184 = (_var183 % train_time_0)
    print(_var184)
    t0_2 = time.time()
    _var185 = global_wrapper(X_test_1)
    pred_0 = clf_2.predict(_var185)
    _var186 = time.time()
    test_time_0 = (_var186 - t0_2)
    _var187 = 'test time:  %0.3fs'
    _var188 = (_var187 % test_time_0)
    print(_var188)
    _var189 = global_wrapper(y_test_0)
    score = metrics.accuracy_score(_var189, pred_0)
    _var190 = 'accuracy:   %0.3f'
    _var191 = (_var190 % score)
    print(_var191)
    _var196 = global_wrapper(y_test_0)
    _var197 = 'weighted'
    (_var192, _var193, _var194, _var195) = precision_recall_fscore_support(_var196, pred_0, average=_var197)
    prec = _var192
    rec = _var193
    f1 = _var194
    _ = _var195
    try:
        feats = clf_2.feature_importances_
        _var198 = np.argsort(feats)
        _var199 = (- 1)
        indices = _var198[::_var199]
        _var200 = 'Feature ranking:'
        print(_var200)
        _var201 = len(feats)
        _var202 = range(_var201)
        for f_4 in _var202:
            _var203 = '%d. feature %d (%f)'
            _var204 = 1
            _var205 = (f_4 + _var204)
            _var206 = indices[f_4]
            _var207 = indices[f_4]
            _var208 = feats[_var207]
            _var209 = (_var205, _var206, _var208)
            _var210 = (_var203 % _var209)
            print(_var210)
    except:
        feats_0 = 0
    _var211 = str(clf_2)
    _var212 = '('
    _var213 = _var211.split(_var212)
    _var214 = 0
    clf_descr = _var213[_var214]
    return (clf_descr, score, prec, rec, f1, feats)
cm = []
_var215 = 15
_var216 = 'auto'
_var217 = (- 1)
_var218 = 0
_var219 = ExtraTreesClassifier(n_estimators=_var215, class_weight=_var216, n_jobs=_var217, random_state=_var218)
_var220 = 'Extra Trees'
clf_3 = _var219
name = _var220
_var221 = '='
_var222 = 80
_var223 = (_var221 * _var222)
print(_var223)
print(name)
_var224 = conf_matrix(clf_3)
cm.append(_var224)
results = []
_var225 = 10
_var226 = (- 1)
_var227 = 'auto'
_var228 = RandomForestClassifier(n_estimators=_var225, n_jobs=_var226, class_weight=_var227)
_var229 = 'Random Forest'
_var230 = (_var228, _var229)
_var231 = 10
_var232 = 'auto'
_var233 = (- 1)
_var234 = 0
_var235 = ExtraTreesClassifier(n_estimators=_var231, class_weight=_var232, n_jobs=_var233, random_state=_var234)
_var236 = 'Extra Trees'
_var237 = (_var235, _var236)
_var238 = (_var230, _var237)
for _var239 in _var238:
    _var242 = 0
    clf_4 = _var239[_var242]
    _var243 = 1
    name_0 = _var239[_var243]
    _var244 = '='
    _var245 = 80
    _var246 = (_var244 * _var245)
    print(_var246)
    print(name_0)
    _var247 = benchmark(clf_4)
    results.append(_var247)
name_1 = __phi__(name_0, name)
clf_5 = __phi__(clf_4, clf_3)
_var248 = X_test_1.columns
_var249 = 1
_var250 = results[_var249]
_var251 = 5
_var252 = _var250[_var251]
_var253 = zip(_var248, _var252)

def _func8(x_4):
    _var254 = 1
    _var255 = x_4[_var254]
    return _var255
sorted(_var253, key=_func8)
results_0 = []
_var256 = 'auto'
_var257 = DecisionTreeClassifier(class_weight=_var256)
_var258 = 'Decision Tree'
_var259 = (_var257, _var258)
_var260 = (- 1)
_var261 = 10
_var262 = 'auto'
_var263 = Perceptron(n_jobs=_var260, n_iter=_var261, class_weight=_var262)
_var264 = 'Perceptron'
_var265 = (_var263, _var264)
_var266 = 'auto'
_var267 = LogisticRegression(class_weight=_var266)
_var268 = 'Logistic'
_var269 = (_var267, _var268)
_var270 = 'auto'
_var271 = (- 1)
_var272 = 1
_var273 = SGDClassifier(class_weight=_var270, n_jobs=_var271, random_state=_var272)
_var274 = 'Stochastic Gradient Descent'
_var275 = (_var273, _var274)
_var276 = (_var259, _var265, _var269, _var275)
for _var277 in _var276:
    _var280 = 0
    clf_6 = _var277[_var280]
    _var281 = 1
    name_2 = _var277[_var281]
    _var282 = '='
    _var283 = 80
    _var284 = (_var282 * _var283)
    print(_var284)
    print(name_2)
    _var285 = benchmark(clf_6)
    results_0.append(_var285)
name_3 = __phi__(name_2, name_1)
clf_7 = __phi__(clf_6, clf_5)
_var286 = '='
_var287 = 80
_var288 = (_var286 * _var287)
print(_var288)
_var289 = 'LinearSVC with L1-based feature selection'
print(_var289)
_var290 = 'feature_selection'
_var291 = 'l1'
_var292 = False
_var293 = 0.001
_var294 = 'auto'
_var295 = LinearSVC(penalty=_var291, dual=_var292, tol=_var293, class_weight=_var294)
_var296 = (_var290, _var295)
_var297 = 'classification'
_var298 = 5
_var299 = RandomForestClassifier(n_estimators=_var298)
_var300 = (_var297, _var299)
_var301 = [_var296, _var300]
_var302 = Pipeline(_var301)
_var303 = benchmark(_var302)
results_0.append(_var303)
_var304 = 'auto'
_var305 = DecisionTreeClassifier(class_weight=_var304)
_var306 = 30
_var307 = 1
_var308 = 1
clf_8 = AdaBoostClassifier(_var305, n_estimators=_var306, learning_rate=_var307, random_state=_var308)
_var309 = benchmark(clf_8)
results_0.append(_var309)
_var310 = 1
_var311 = results_0[_var310]
_var312 = 6
cm_array = _var311[_var312]

def plot_confusion_matrix(cm_0, title='Actual versus Predicted', cmap=plt.cm.Blues):
    _var313 = cm_0.squeeze()
    _var314 = 'nearest'
    plt.imshow(_var313, interpolation=_var314, cmap=cmap)
    _var315 = 15
    plt.title(title, size=_var315)
    plt.colorbar()
    _var316 = 37
    tick_marks = np.arange(_var316)
    _var317 = global_wrapper(df_sub)
    _var318 = _var317.n1
    _var319 = _var318.unique()
    _var320 = sorted(_var319)
    _var321 = 90
    plt.xticks(tick_marks, _var320, rotation=_var321)
    _var322 = global_wrapper(df_sub)
    _var323 = _var322.n1
    _var324 = _var323.unique()
    _var325 = sorted(_var324)
    plt.yticks(tick_marks, _var325)
    plt.tight_layout()
    _var326 = 'True label'
    plt.ylabel(_var326)
    _var327 = 'Predicted label'
    plt.xlabel(_var327)
_var328 = 2
np.set_printoptions(precision=_var328)
_var329 = 'Actual versus Predicted Destination Neighborhoods (without normalization)'
print(_var329)
print(cm_array)
_var330 = 10
_var331 = 6
_var332 = (_var330, _var331)
plt.figure(figsize=_var332)
plot_confusion_matrix(cm_array)
_var333 = 'float'
_var334 = cm_array.astype(_var333)
_var335 = 1
_var336 = cm_array.sum(axis=_var335)
_var337 = np.newaxis
_var338 = _var336[:, _var337]
cm_normalized = (_var334 / _var338)
_var339 = 'Normalized confusion matrix'
print(_var339)
print(cm_normalized)
_var340 = 14
_var341 = 10
_var342 = (_var340, _var341)
plt.figure(figsize=_var342)
_var343 = 'Normalized confusion matrix'
plot_confusion_matrix(cm_normalized, title=_var343)
import operator
_var344 = 'n2_'
_var345 = df_0.groupby(_var344)
_var346 = _var345.same_station
_var347 = _var346.value_counts()
t_2 = pd.DataFrame(_var347)
d = {}
_var348 = df_0.n2_
_var349 = _var348.unique()
for n in _var349:
    _var350 = t_2.loc
    _var351 = _var350[n]
    _var352 = _var351.iloc
    _var353 = 1
    _var354 = _var352[_var353]
    _var355 = _var354.values
    _var356 = t_2.loc
    _var357 = _var356[n]
    _var358 = _var357.iloc
    _var359 = 0
    _var360 = _var358[_var359]
    _var361 = _var360.values
    _var362 = t_2.loc
    _var363 = _var362[n]
    _var364 = _var363.iloc
    _var365 = 1
    _var366 = _var364[_var365]
    _var367 = _var366.values
    _var368 = (_var361 + _var367)
    _var369 = (_var355 / _var368)
    d_0 = set_index_wrapper(d, n, _var369)
d_1 = __phi__(d_0, d)
_var370 = d_1.items()
_var371 = list(_var370)
_var372 = 1
_var373 = operator.itemgetter(_var372)
sorted_x = sorted(_var371, key=_var373)
sorted_x.reverse()
df_stations = pd.DataFrame(sorted_x)
_var374 = ['Neighborhood', 'Removal rate']
df_stations_0 = set_field_wrapper(df_stations, 'columns', _var374)
_var375 = 'bikeid'
_var376 = df_0[_var375]
_var377 = 21415
_var378 = (_var376 == _var377)
_var379 = df_0[_var378]
_var380 = ['bikeid', 'stoptime_', 'end_station_id_', 'n2_', 'starttime', 'start_station_id', 'n1', 'same_station']
bike_21415 = _var379[_var380]
bike_21415
