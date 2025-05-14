

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
from sklearn.preprocessing import Imputer, LabelEncoder, OneHotEncoder
from visualization import visualizeCorrelations
_var0 = 'Project Train Dataset.csv'
df = pd.read_csv(_var0)
_var1 = df.columns
_var2 = _var1.str
_var3 = '"'
_var4 = ''
_var5 = _var2.replace(_var3, _var4)
_var6 = _var5.str
_var7 = ','
_var8 = ';'
_var9 = _var6.replace(_var7, _var8)
df_0 = set_field_wrapper(df, 'columns', _var9)
_var10 = df_0.iloc
_var11 = 0
_var12 = df_0.iloc
_var13 = 0
_var14 = _var12[:, _var13]
_var15 = _var14.str
_var16 = '"'
_var17 = ''
_var18 = _var15.replace(_var16, _var17)
_var19 = _var18.str
_var20 = ','
_var21 = ';'
_var22 = _var19.replace(_var20, _var21)
_var10_0 = set_index_wrapper(_var10, (slice(None, None, None), _var11), _var22)
_var23 = 'train.csv'
_var24 = False
df_0.to_csv(_var23, index=_var24)
_var25 = 'train.csv'
_var26 = ';'
dataset = pd.read_csv(_var25, sep=_var26)
visualizeCorrelations(dataset)
_var27 = 'SEX'
_var28 = 'SEX'
_var29 = dataset[_var28]
_var30 = 'F'
_var31 = 0
_var32 = _var29.fillna(_var30, axis=_var31)
dataset_0 = set_index_wrapper(dataset, _var27, _var32)
_var33 = 'EDUCATION'
_var34 = 'EDUCATION'
_var35 = dataset_0[_var34]
_var36 = 'university'
_var37 = 0
_var38 = _var35.fillna(_var36, axis=_var37)
dataset_1 = set_index_wrapper(dataset_0, _var33, _var38)
_var39 = 'MARRIAGE'
_var40 = 'MARRIAGE'
_var41 = dataset_1[_var40]
_var42 = 'single'
_var43 = 0
_var44 = _var41.fillna(_var42, axis=_var43)
dataset_2 = set_index_wrapper(dataset_1, _var39, _var44)
_var45 = dataset_2.loc
_var46 = 'PAY_DEC'
_var47 = 'PAY_JUL'
_var48 = dataset_2.loc
_var49 = 'PAY_DEC'
_var50 = 'PAY_JUL'
_var51 = _var48[:, _var49:_var50]
_var52 = (- 1)
_var53 = (- 2)
_var54 = [_var52, _var53]
_var55 = 0
_var56 = _var51.replace(to_replace=_var54, value=_var55)
_var45_0 = set_index_wrapper(_var45, (slice(None, None, None), slice(_var46, _var47, None)), _var56)
_var57 = dataset_2.loc
_var58 = 'BILL_AMT_DEC'
_var59 = 'BILL_AMT_JUL'
_var60 = _var57[:, _var58:_var59]
_var61 = 1
billMean = _var60.mean(axis=_var61)
_var62 = dataset_2.loc
_var63 = 'PAY_AMT_DEC'
_var64 = 'PAY_AMT_JUL'
_var65 = _var62[:, _var63:_var64]
_var66 = 1
payMean = _var65.mean(axis=_var66)
_var67 = 'BILL_AMT_DEC'
dataset_3 = set_index_wrapper(dataset_2, _var67, billMean)
_var68 = 'PAY_AMT_DEC'
dataset_4 = set_index_wrapper(dataset_3, _var68, payMean)
_var69 = ['BILL_AMT_NOV', 'BILL_AMT_OCT', 'BILL_AMT_SEP', 'BILL_AMT_AUG', 'BILL_AMT_JUL']
_var70 = 1
dataset_5 = dataset_4.drop(_var69, _var70)
_var71 = ['PAY_AMT_NOV', 'PAY_AMT_OCT', 'PAY_AMT_SEP', 'PAY_AMT_AUG', 'PAY_AMT_JUL']
_var72 = 1
dataset_6 = dataset_5.drop(_var71, _var72)
from datetime import date, datetime
import time

def calculate_age(born):
    _var73 = isinstance(born, float)
    if _var73:
        return born
    _var74 = '%d/%m/%Y'
    born_0 = datetime.strptime(born, _var74)
    today = date.today()
    _var75 = today.year
    _var76 = born_0.year
    _var77 = (_var75 - _var76)
    _var78 = today.month
    _var79 = today.day
    _var80 = (_var78, _var79)
    _var81 = born_0.month
    _var82 = born_0.day
    _var83 = (_var81, _var82)
    _var84 = (_var80 < _var83)
    _var85 = (_var77 - _var84)
    return _var85
_var86 = 'BIRTH_DATE'
_var87 = 'BIRTH_DATE'
_var88 = dataset_6[_var87]

def _func0(x):
    _var89 = calculate_age(x)
    return _var89
_var90 = _var88.map(_func0)
dataset_7 = set_index_wrapper(dataset_6, _var86, _var90)
_var91 = np.nan
_var92 = 'median'
_var93 = 1
imputer = Imputer(missing_values=_var91, strategy=_var92, axis=_var93)
_var94 = dataset_7.iloc
_var95 = 5
_var96 = dataset_7.iloc
_var97 = 5
_var98 = _var96[:, _var97]
_var99 = imputer.fit_transform(_var98)
_var100 = _var99.flatten()
_var94_0 = set_index_wrapper(_var94, (slice(None, None, None), _var95), _var100)
labelEncoder = LabelEncoder()
_var101 = dataset_7.iloc
_var102 = 2
_var103 = dataset_7.iloc
_var104 = 2
_var105 = _var103[:, _var104]
_var106 = labelEncoder.fit_transform(_var105)
_var107 = _var106.flatten()
_var101_0 = set_index_wrapper(_var101, (slice(None, None, None), _var102), _var107)
_var108 = dataset_7.iloc
_var109 = 3
_var110 = dataset_7.iloc
_var111 = 3
_var112 = _var110[:, _var111]
_var113 = labelEncoder.fit_transform(_var112)
_var114 = _var113.flatten()
_var108_0 = set_index_wrapper(_var108, (slice(None, None, None), _var109), _var114)
_var115 = dataset_7.iloc
_var116 = 4
_var117 = dataset_7.iloc
_var118 = 4
_var119 = _var117[:, _var118]
_var120 = labelEncoder.fit_transform(_var119)
_var121 = _var120.flatten()
_var115_0 = set_index_wrapper(_var115, (slice(None, None, None), _var116), _var121)
_var122 = [2, 3, 4]
oneHotEncoder = OneHotEncoder(categorical_features=_var122)
_var123 = dataset_7.values
_var124 = oneHotEncoder.fit_transform(_var123)
X = _var124.toarray()
_var125 = [0, 2, 6]
_var126 = 1
X_0 = np.delete(X, _var125, _var126)
_var127 = 6
_var128 = 1
X_1 = np.delete(X_0, _var127, _var128)
_var129 = (- 1)
X_train = X_1[:, :_var129]
_var130 = (- 1)
y_train = X_1[:, _var130]
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train_0 = sc_X.fit_transform(X_train)
from imblearn.over_sampling import RandomOverSampler
_var131 = 42
ros = RandomOverSampler(random_state=_var131)
(_var132, _var133) = ros.fit_sample(X_train_0, y_train)
X_train_1 = _var132
y_train_0 = _var133
_var134 = 'Oversampling completed, new shape of data: '
_var135 = X_train_1.shape
_var136 = (_var134, _var135)
print(_var136)
from sklearn.svm import SVC, NuSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier, VotingClassifier, IsolationForest, BaggingClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
_var137 = 0
_var138 = 0.05
_var139 = 'l2'
_var140 = LogisticRegression(random_state=_var137, C=_var138, penalty=_var139)
_var141 = 1000
_var142 = 'entropy'
_var143 = 0
_var144 = (- 1)
_var145 = RandomForestClassifier(n_estimators=_var141, criterion=_var142, random_state=_var143, n_jobs=_var144)
_var146 = GaussianNB()
_var147 = 10
_var148 = 'minkowski'
_var149 = 2
_var150 = KNeighborsClassifier(n_neighbors=_var147, metric=_var148, p=_var149)
_var151 = 'rbf'
_var152 = 0
_var153 = SVC(kernel=_var151, random_state=_var152)
_var154 = 0
_var155 = DecisionTreeClassifier(random_state=_var154)
_var156 = 100
_var157 = 0
_var158 = 'gini'
_var159 = 0
_var160 = DecisionTreeClassifier(criterion=_var158, random_state=_var159)
_var161 = AdaBoostClassifier(n_estimators=_var156, random_state=_var157, base_estimator=_var160)
_var162 = 'lbfgs'
_var163 = 1e-05
_var164 = 5
_var165 = 2
_var166 = (_var164, _var165)
_var167 = 1
_var168 = MLPClassifier(solver=_var162, alpha=_var163, hidden_layer_sizes=_var166, random_state=_var167)
_var169 = 100
_var170 = 9
_var171 = 0.02
_var172 = 1e-07
_var173 = 1e-05
_var174 = 0.5
_var175 = XGBClassifier(n_estimators=_var169, max_depth=_var170, learning_rate=_var171, gamma=_var172, reg_lambda=_var173, subsample=_var174)
_var176 = 50
_var177 = None
_var178 = 0
_var179 = (- 1)
_var180 = ExtraTreesClassifier(n_estimators=_var176, max_depth=_var177, random_state=_var178, n_jobs=_var179)
_var181 = 100
_var182 = 1e-08
_var183 = 100
_var184 = 0.02
_var185 = 0.7
_var186 = XGBClassifier(n_estimators=_var181, gamma=_var182, reg_lambda=_var183, learning_rate=_var184, subsample=_var185)
_var187 = 50
_var188 = 0.5
_var189 = 0.5
_var190 = (- 1)
_var191 = 0
_var192 = BaggingClassifier(_var186, n_estimators=_var187, max_samples=_var188, max_features=_var189, n_jobs=_var190, random_state=_var191)
models = {'LogisticRegression': _var140, 'RandomForest': _var145, 'NaiveBayes': _var146, 'KNN': _var150, 'KernelSVM': _var153, 'DecisionTree': _var155, 'AdaBoost': _var161, 'MLPClassifier': _var168, 'XGBClassifier': _var175, 'ExtraTreesClassifier': _var180, 'BaggingClassifier': _var192}
_var193 = 'BaggingClassifier'
classifier = models[_var193]
start = time.time()
classifier_0 = classifier.fit(X_train_1, y_train_0)
_var194 = 'Fit completed in %s seconds'
_var195 = time.time()
_var196 = (_var195 - start)
_var197 = (_var194 % _var196)
print(_var197)
_var198 = 'Project Test Dataset.csv'
_var199 = ';'
test_dataset = pd.read_csv(_var198, sep=_var199)
_var200 = 'SEX'
_var201 = 'SEX'
_var202 = test_dataset[_var201]
_var203 = 'F'
_var204 = 0
_var205 = _var202.fillna(_var203, axis=_var204)
test_dataset_0 = set_index_wrapper(test_dataset, _var200, _var205)
_var206 = 'EDUCATION'
_var207 = 'EDUCATION'
_var208 = test_dataset_0[_var207]
_var209 = 'university'
_var210 = 0
_var211 = _var208.fillna(_var209, axis=_var210)
test_dataset_1 = set_index_wrapper(test_dataset_0, _var206, _var211)
_var212 = 'MARRIAGE'
_var213 = 'MARRIAGE'
_var214 = test_dataset_1[_var213]
_var215 = 'single'
_var216 = 0
_var217 = _var214.fillna(_var215, axis=_var216)
test_dataset_2 = set_index_wrapper(test_dataset_1, _var212, _var217)
_var218 = test_dataset_2.loc
_var219 = 'PAY_DEC'
_var220 = 'PAY_JUL'
_var221 = test_dataset_2.loc
_var222 = 'PAY_DEC'
_var223 = 'PAY_JUL'
_var224 = _var221[:, _var222:_var223]
_var225 = (- 1)
_var226 = (- 2)
_var227 = [_var225, _var226]
_var228 = 0
_var229 = _var224.replace(to_replace=_var227, value=_var228)
_var218_0 = set_index_wrapper(_var218, (slice(None, None, None), slice(_var219, _var220, None)), _var229)
_var230 = test_dataset_2.loc
_var231 = 'BILL_AMT_DEC'
_var232 = 'BILL_AMT_JUL'
_var233 = _var230[:, _var231:_var232]
_var234 = 1
billMean_0 = _var233.mean(axis=_var234)
_var235 = test_dataset_2.loc
_var236 = 'PAY_AMT_DEC'
_var237 = 'PAY_AMT_JUL'
_var238 = _var235[:, _var236:_var237]
_var239 = 1
payMean_0 = _var238.mean(axis=_var239)
_var240 = 'BILL_AMT_DEC'
test_dataset_3 = set_index_wrapper(test_dataset_2, _var240, billMean_0)
_var241 = 'PAY_AMT_DEC'
test_dataset_4 = set_index_wrapper(test_dataset_3, _var241, payMean_0)
_var242 = ['BILL_AMT_NOV', 'BILL_AMT_OCT', 'BILL_AMT_SEP', 'BILL_AMT_AUG', 'BILL_AMT_JUL']
_var243 = 1
test_dataset_5 = test_dataset_4.drop(_var242, _var243)
_var244 = ['PAY_AMT_NOV', 'PAY_AMT_OCT', 'PAY_AMT_SEP', 'PAY_AMT_AUG', 'PAY_AMT_JUL']
_var245 = 1
test_dataset_6 = test_dataset_5.drop(_var244, _var245)

def calculate_age_test(born_1):
    _var246 = isinstance(born_1, float)
    if _var246:
        return born_1
    _var247 = len(born_1)
    _var248 = 19
    _var249 = (_var247 < _var248)
    if _var249:
        _var250 = '%d/%m/%Y'
        born_2 = datetime.strptime(born_1, _var250)
    else:
        _var251 = 10
        _var252 = born_1[:_var251]
        _var253 = '%Y-%m-%d'
        born_3 = datetime.strptime(_var252, _var253)
    born_4 = __phi__(born_2, born_3)
    today_0 = date.today()
    _var254 = today_0.year
    _var255 = born_4.year
    _var256 = (_var254 - _var255)
    _var257 = today_0.month
    _var258 = today_0.day
    _var259 = (_var257, _var258)
    _var260 = born_4.month
    _var261 = born_4.day
    _var262 = (_var260, _var261)
    _var263 = (_var259 < _var262)
    _var264 = (_var256 - _var263)
    return _var264
_var265 = 'BIRTH_DATE'
_var266 = 'BIRTH_DATE'
_var267 = test_dataset_6[_var266]

def _func1(x_0):
    _var268 = calculate_age_test(x_0)
    return _var268
_var269 = _var267.map(_func1)
test_dataset_7 = set_index_wrapper(test_dataset_6, _var265, _var269)
_var270 = np.nan
_var271 = 'median'
_var272 = 1
imputer_0 = Imputer(missing_values=_var270, strategy=_var271, axis=_var272)
_var273 = test_dataset_7.iloc
_var274 = 5
_var275 = test_dataset_7.iloc
_var276 = 5
_var277 = _var275[:, _var276]
_var278 = imputer_0.fit_transform(_var277)
_var279 = _var278.flatten()
_var273_0 = set_index_wrapper(_var273, (slice(None, None, None), _var274), _var279)
labelEncoder_0 = LabelEncoder()
_var280 = test_dataset_7.iloc
_var281 = 2
_var282 = test_dataset_7.iloc
_var283 = 2
_var284 = _var282[:, _var283]
_var285 = labelEncoder_0.fit_transform(_var284)
_var286 = _var285.flatten()
_var280_0 = set_index_wrapper(_var280, (slice(None, None, None), _var281), _var286)
_var287 = test_dataset_7.iloc
_var288 = 3
_var289 = test_dataset_7.iloc
_var290 = 3
_var291 = _var289[:, _var290]
_var292 = labelEncoder_0.fit_transform(_var291)
_var293 = _var292.flatten()
_var287_0 = set_index_wrapper(_var287, (slice(None, None, None), _var288), _var293)
_var294 = test_dataset_7.iloc
_var295 = 4
_var296 = test_dataset_7.iloc
_var297 = 4
_var298 = _var296[:, _var297]
_var299 = labelEncoder_0.fit_transform(_var298)
_var300 = _var299.flatten()
_var294_0 = set_index_wrapper(_var294, (slice(None, None, None), _var295), _var300)
_var301 = ['DEFAULT PAYMENT JAN']
_var302 = 1
test_dataset_8 = test_dataset_7.drop(_var301, _var302)
_var303 = [2, 3, 4]
oneHotEncoder_0 = OneHotEncoder(categorical_features=_var303)
_var304 = test_dataset_8.values
_var305 = oneHotEncoder_0.fit_transform(_var304)
X_test = _var305.toarray()
_var306 = [0, 2, 6]
_var307 = 1
X_test_0 = np.delete(X_test, _var306, _var307)
_var308 = 6
_var309 = 1
X_test_1 = np.delete(X_test_0, _var308, _var309)
X_test_2 = sc_X.fit_transform(X_test_1)
y_pred = classifier_0.predict(X_test_2)
_var310 = 'Prediction completed'
print(_var310)
_var311 = 'test.csv'
_var312 = '%d'
_var313 = 'DEFAULT PAYMENT JAN'
np.savetxt(_var311, y_pred, fmt=_var312, header=_var313)
