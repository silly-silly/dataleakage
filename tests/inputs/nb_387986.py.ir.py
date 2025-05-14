

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
_var0 = 'homework2_data.csv'
_var1 = 99999
df = pd.read_csv(_var0, na_values=_var1)
df.head()
_var2 = df.uf17
no_rent = _var2.isnull()
_var3 = (~ no_rent)
df_rent = df[_var3]
rent = df_rent.uf17
_var4 = 'uf17'
_var5 = 1
data = df_rent.drop(_var4, axis=_var5)
_var6 = 'race1'
_var7 = 1
data_0 = data.drop(_var6, axis=_var7)
import matplotlib.pyplot as plt
_var8 = get_ipython()
_var9 = 'matplotlib'
_var10 = 'inline'
_var8.run_line_magic(_var9, _var10)
_var11 = 30
rent.hist(bins=_var11)
_var12 = 0
_var13 = data_0.var(axis=_var12)
_var14 = 0
non_const_columns = (_var13 > _var14)
_var15 = 0
_var16 = data_0.var(axis=_var15)
_var16.head()
_var17 = data_0.columns
_var18 = _var17[non_const_columns]
data_ = data_0[_var18]
data_.shape
from sklearn.preprocessing import Imputer
_var19 = 'median'
_var20 = Imputer(strategy=_var19)
_var21 = data_.values
X = _var20.fit_transform(_var21)
X.shape
_var22 = rent.values
_var22.shape
from sklearn.feature_selection import f_regression
_var25 = rent.values
(_var23, _var24) = f_regression(X, _var25)
F = _var23
p = _var24
import numpy as np
_var26 = 10
_var27 = 50
_var28 = (_var26, _var27)
plt.figure(figsize=_var28)
_var29 = len(F)
_var30 = np.arange(_var29)
_var31 = 1
_var32 = (F + _var31)
_var33 = np.log(_var32)
plt.barh(_var30, _var33)
_var34 = len(F)
_var35 = np.arange(_var34)
_var36 = data_.columns
plt.yticks(_var35, _var36)
_var37 = 'homework2_variable_description.xls'
_var38 = 4
raw_variables = pd.read_excel(_var37, skiprows=_var38)
_var39 = 'Variable Name'
_var40 = raw_variables[_var39]
_var41 = 'Variable Name'
_var42 = (_var40 != _var41)
raw_variables_0 = raw_variables[_var42]
raw_variables_0.head()
_var43 = ['Variable Name']
variables = raw_variables_0.dropna(subset=_var43)
variables.head()
_var44 = variables.iterrows()
for _var45 in _var44:
    _var48 = 0
    i = _var45[_var48]
    _var49 = 1
    row = _var45[_var49]
    _var50 = 1
    _var51 = (i - _var50)
    _var52 = variables.index
    _var53 = (_var51 in _var52)
    if _var53:
        _var54 = variables.loc
        _var55 = 1
        _var56 = (i - _var55)
        row_above = _var54[_var56]
        _var57 = 'Variable Name'
        _var58 = row_above[_var57]
        _var59 = 'REC'
        _var60 = _var58.startswith(_var59)
        if _var60:
            _var61 = variables.loc
            _var62 = _var61[i]
            _var63 = 'Item Name'
            _var64 = variables.loc
            _var65 = 1
            _var66 = (i - _var65)
            _var67 = _var64[_var66]
            _var68 = 'Item Name'
            _var69 = _var67[_var68]
            _var62_0 = set_index_wrapper(_var62, _var63, _var69)
_var70 = 'Item Name'
_var71 = variables[_var70]
mask = _var71.isnull()
variables_0 = variables.copy()
_var72 = 'Item Name'
_var73 = variables_0[_var72]
_var74 = 'Code and Description'
_var75 = variables_0[_var74]
_var76 = _var75[mask]
_var73_0 = set_index_wrapper(_var73, mask, _var76)
_var77 = 'Variable Name'
_var78 = 'Variable Name'
_var79 = variables_0[_var78]
_var80 = _var79.str
_var81 = ','
_var82 = _var80.strip(_var81)
_var83 = 'SEX/HHR2'
_var84 = 'HHR2'
_var85 = _var82.replace(_var83, _var84)
variables_1 = set_index_wrapper(variables_0, _var77, _var85)
_var86 = 'Variable Name'
_var87 = variables_1[_var86]
_var88 = _var87.str
_var89 = _var88.lower()
variables_2 = variables_1.set_index(_var89)
_var90 = variables_2.loc
_var91 = 'uf43'
_var92 = _var90[_var91]
_var93 = 'Item Name'
_var94 = variables_2.loc
_var95 = 'hhr3t'
_var96 = _var94[_var95]
_var97 = 'Item Name'
_var98 = _var96[_var97]
_var92_0 = set_index_wrapper(_var92, _var93, _var98)
variables_2.head()
_var99 = 'Item Name'
item_name = variables_2[_var99]
item_name.head()
_var100 = ['Source Code']
_var101 = raw_variables_0.dropna(subset=_var100)
source_codes = _var101.copy()
_var102 = 'Source Code'
_var103 = 'Source Code'
_var104 = source_codes[_var103]
_var105 = 'str'
_var106 = _var104.astype(_var105)
_var107 = _var106.str
_var108 = '0'
_var109 = _var107.lstrip(_var108)
source_codes_0 = set_index_wrapper(source_codes, _var102, _var109)
_var110 = ['Source Code']
source_codes_1 = source_codes_0.drop_duplicates(subset=_var110)
_var111 = 'sc'
_var112 = 'Source Code'
_var113 = source_codes_1[_var112]
_var114 = (_var111 + _var113)
sc_reindex = source_codes_1.set_index(_var114)
_var115 = 'Item Name'
source_code_series = sc_reindex[_var115]
feature_mapping = {}
_var116 = data_.columns
for c in _var116:
    _var117 = item_name.index
    _var118 = (c in _var117)
    if _var118:
        _var119 = item_name.loc
        _var120 = _var119[c]
        feature_mapping_0 = set_index_wrapper(feature_mapping, c, _var120)
    else:
        _var121 = source_code_series.index
        _var122 = (c in _var121)
        if _var122:
            _var123 = source_code_series.loc
            _var124 = _var123[c]
            feature_mapping_1 = set_index_wrapper(feature_mapping, c, _var124)
        else:
            print(c)
        feature_mapping_2 = __phi__(feature_mapping_1, feature_mapping)
    feature_mapping_3 = __phi__(feature_mapping_0, feature_mapping_2)
feature_mapping_4 = __phi__(feature_mapping_3, feature_mapping)
_var125 = 'seqno'
_var126 = 'seqno'
feature_mapping_5 = set_index_wrapper(feature_mapping_4, _var125, _var126)
_var127 = 1
data_desc = data_.rename_axis(item_name, axis=_var127)
_var128 = 1
data_desc_0 = data_desc.rename_axis(source_code_series, axis=_var128)
_var129 = 10
_var130 = 50
_var131 = (_var129, _var130)
plt.figure(figsize=_var131)
_var132 = len(F)
_var133 = np.arange(_var132)
_var134 = 1
_var135 = (F + _var134)
_var136 = np.log(_var135)
plt.barh(_var133, _var136)
_var137 = len(F)
_var138 = np.arange(_var137)
_var139 = data_desc_0.columns
plt.yticks(_var138, _var139)
from sklearn.feature_selection import mutual_info_regression
_var140 = rent.values
mi = mutual_info_regression(X, _var140)
inds = np.argsort(mi)
_var141 = 5
_var142 = 30
_var143 = (_var141, _var142)
plt.figure(figsize=_var143)
_var144 = len(mi)
_var145 = np.arange(_var144)
_var146 = mi[inds]
_var147 = 1
_var148 = (_var146 + _var147)
_var149 = np.log(_var148)
plt.barh(_var145, _var149)
_var150 = len(mi)
_var151 = np.arange(_var150)
_var152 = [feature_mapping[x] for x in data_.columns[inds]]
plt.yticks(_var151, _var152)
_var153 = data_desc_0.columns
_var154 = 'Kitchen Facilities Functioning'
_var155 = (_var153 == _var154)
np.where(_var155)
_var156 = data_.columns
_var157 = 58
_var158 = 1
_var159 = (_var157 + _var158)
non_renter_columns = _var156[:_var159]
non_renter_columns_0 = [i for i in non_renter_columns if (('Householder' not in feature_mapping[i]) and ('Number of Persons from' not in feature_mapping[i]) and ('Origin' not in feature_mapping[i]))]
_var160 = ['new_csr']
non_renter_columns_1 = (non_renter_columns_0 + _var160)
data_nr = data_[non_renter_columns_1]
_var161 = 'median'
_var162 = Imputer(strategy=_var161)
_var163 = data_nr.values
X_0 = _var162.fit_transform(_var163)
_var166 = rent.values
(_var164, _var165) = f_regression(X_0, _var166)
F_0 = _var164
p_0 = _var165
_var167 = 10
_var168 = 10
_var169 = (_var167, _var168)
plt.figure(figsize=_var169)
_var170 = len(F_0)
_var171 = np.arange(_var170)
_var172 = 1
_var173 = (F_0 + _var172)
_var174 = np.log(_var173)
plt.barh(_var171, _var174)
_var175 = len(F_0)
_var176 = np.arange(_var175)
_var177 = [feature_mapping[x] for x in data_nr.columns]
plt.yticks(_var176, _var177)
from sklearn.feature_selection import mutual_info_regression
_var178 = rent.values
mi_0 = mutual_info_regression(X_0, _var178)
_var179 = 10
_var180 = 10
_var181 = (_var179, _var180)
plt.figure(figsize=_var181)
_var182 = len(mi_0)
_var183 = np.arange(_var182)
plt.barh(_var183, mi_0)
_var184 = len(mi_0)
_var185 = np.arange(_var184)
_var186 = [feature_mapping[x] for x in data_nr.columns]
plt.yticks(_var185, _var186)
from sklearn.feature_selection import SelectPercentile
_var187 = 50
select = SelectPercentile(mutual_info_regression, percentile=_var187)
_var188 = rent.values
select_0 = select.fit(X_0, _var188)
feature_names = [feature_mapping[x] for x in data_nr.columns[select.get_support()]]
feature_names
X_selected = select_0.transform(X_0)
len(feature_names)
y = rent.values
_var191 = 6
_var192 = 4
_var193 = 20
_var194 = 10
_var195 = (_var193, _var194)
(_var189, _var190) = plt.subplots(_var191, _var192, figsize=_var195)
fig = _var189
axes = _var190
_var196 = axes.ravel()
_var197 = zip(feature_names, _var196)
_var198 = enumerate(_var197)
for _var199 in _var198:
    _var202 = 0
    i_0 = _var199[_var202]
    _var205 = 1
    _var206 = _var199[_var205]
    _var207 = 0
    name = _var206[_var207]
    _var208 = 1
    ax = _var206[_var208]
    _var209 = global_wrapper(ax)
    _var210 = X_selected[:, i_0]
    _var211 = 0.1
    _var209.scatter(_var210, y, alpha=_var211)
    _var212 = global_wrapper(ax)
    _var213 = '{} : {}'
    _var214 = _var213.format(i_0, name)
    _var212.set_title(_var214)
i_1 = __phi__(i_0, i)
plt.tight_layout()
X_0.shape
from sklearn.model_selection import train_test_split
_var219 = 0
(_var215, _var216, _var217, _var218) = train_test_split(X_selected, y, random_state=_var219)
X_trainval = _var215
X_test = _var216
y_trainval = _var217
y_test = _var218
_var224 = 0
(_var220, _var221, _var222, _var223) = train_test_split(X_trainval, y_trainval, random_state=_var224)
X_train = _var220
X_val = _var221
y_train = _var222
y_val = _var223
from sklearn.ensemble import RandomForestRegressor
_var225 = 100
_var226 = RandomForestRegressor(n_estimators=_var225)
rf = _var226.fit(X_train, y_train)
_var226_0 = rf
rf.score(X_val, y_val)
_var227 = X_train.shape
_var228 = 1
_var229 = _var227[_var228]
_var230 = range(_var229)
_var231 = list(_var230)
_var232 = rf.feature_importances_
plt.barh(_var231, _var232)
_var233 = X_train.shape
_var234 = 1
_var235 = _var233[_var234]
_var236 = range(_var235)
_var237 = list(_var236)
plt.yticks(_var237, feature_names)
from sklearn.linear_model import LinearRegression
_var238 = LinearRegression()
_var239 = _var238.fit(X_train, y_train)
_var239.score(X_val, y_val)
_var240 = LinearRegression()
lr = _var240.fit(X_train, y_train)
_var240_0 = lr
lr.score(X_val, y_val)
_var241 = lr.predict(X_val)
_var242 = 0.1
plt.scatter(y_val, _var241, alpha=_var242)
from sklearn.preprocessing import OneHotEncoder
_var243 = OneHotEncoder()
ohe = _var243.fit(X_train)
_var243_0 = ohe
from sklearn.pipeline import make_pipeline
_var244 = OneHotEncoder()
_var245 = LinearRegression()
ohe_pipe = make_pipeline(_var244, _var245)
_var246 = ohe_pipe.fit(X_train, y_train)
_var246.score(X_val, y_val)
feature_names
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import RidgeCV, LassoLarsCV, Lars, Ridge, LassoLars
from sklearn.feature_selection import VarianceThreshold
_var247 = False
_var248 = OneHotEncoder(sparse=_var247)
_var249 = True
_var250 = PolynomialFeatures(interaction_only=_var249)
_var251 = VarianceThreshold()
_var252 = RidgeCV()
ohe_interactions_pipe = make_pipeline(_var248, _var250, _var251, _var252)
_var253 = ohe_interactions_pipe.fit(X_train, y_train)
_var253.score(X_val, y_val)
_var254 = ohe_interactions_pipe.predict(X_val)
_var255 = 0.1
plt.scatter(y_val, _var254, alpha=_var255)
_var256 = plt.gca()
_var257 = 'equal'
_var256.set_aspect(_var257)
_var258 = [0, 8000]
_var259 = [0, 8000]
_var260 = 'k'
plt.plot(_var258, _var259, c=_var260)
_var261 = ohe_interactions_pipe.predict(X_val)
_var262 = (y_val - _var261)
_var263 = 100
plt.hist(_var262, bins=_var263)
from sklearn.feature_selection import VarianceThreshold
_var264 = False
_var265 = OneHotEncoder(sparse=_var264)
_var266 = False
_var267 = True
_var268 = PolynomialFeatures(include_bias=_var266, interaction_only=_var267)
_var269 = VarianceThreshold()
ohe_interactions_trans = make_pipeline(_var265, _var268, _var269)
X_expanded = ohe_interactions_trans.fit_transform(X_train)
X_expanded.shape
_var270 = 50
plt.hist(y_train, bins=_var270)
y_train.max()
_var271 = y_train.max()
_var272 = (y_train == _var271)
_var273 = np.sum(_var272)
_var274 = len(y_train)
(_var273 / _var274)
_var275 = 7000
_var276 = (y > _var275)
y[_var276]
_var277 = 7999
_var278 = (y < _var277)
X_selected_ = X_selected[_var278]
_var279 = 7999
_var280 = (y < _var279)
y_ = y[_var280]
_var285 = 0
(_var281, _var282, _var283, _var284) = train_test_split(X_selected_, y_, random_state=_var285)
X_trainval_0 = _var281
X_test_0 = _var282
y_trainval_0 = _var283
y_test_0 = _var284
_var290 = 0
(_var286, _var287, _var288, _var289) = train_test_split(X_trainval_0, y_trainval_0, random_state=_var290)
X_train_0 = _var286
X_val_0 = _var287
y_train_0 = _var288
y_val_0 = _var289
_var291 = 50
plt.hist(y_train_0, bins=_var291)
_var292 = False
_var293 = OneHotEncoder(sparse=_var292)
_var294 = True
_var295 = PolynomialFeatures(interaction_only=_var294)
_var296 = VarianceThreshold()
_var297 = RidgeCV()
ohe_interactions_pipe_0 = make_pipeline(_var293, _var295, _var296, _var297)
_var298 = ohe_interactions_pipe_0.fit(X_train_0, y_train_0)
_var298.score(X_val_0, y_val_0)
_var299 = 10
_var300 = 10
_var301 = (_var299, _var300)
plt.figure(figsize=_var301)
_var302 = ohe_interactions_pipe_0.predict(X_val_0)
_var303 = 0.1
plt.scatter(y_val_0, _var302, alpha=_var303)
_var304 = plt.gca()
_var305 = 'equal'
_var304.set_aspect(_var305)
_var306 = y_val_0.max()
_var307 = [0, _var306]
_var308 = y_val_0.max()
_var309 = [0, _var308]
_var310 = 'k'
plt.plot(_var307, _var309, c=_var310)
from scipy.stats import boxcox
(_var311, _var312) = boxcox(y_train_0)
y_train_bc = _var311
l = _var312
l
_var313 = 50
plt.hist(y_train_bc, bins=_var313)
_var314 = 1
(_var314 / l)
_var315 = 0.27
_var316 = (y_train_0 ** _var315)
_var317 = 50
plt.hist(_var316, bins=_var317)
_var318 = False
_var319 = OneHotEncoder(sparse=_var318)
_var320 = True
_var321 = PolynomialFeatures(interaction_only=_var320)
_var322 = VarianceThreshold()
_var323 = RidgeCV()
ohe_interactions_pipe_1 = make_pipeline(_var319, _var321, _var322, _var323)
_var324 = 0.27
_var325 = (y_train_0 ** _var324)
_var326 = ohe_interactions_pipe_1.fit(X_train_0, _var325)
_var327 = 0.27
_var328 = (y_val_0 ** _var327)
_var326.score(X_val_0, _var328)
from sklearn.metrics import r2_score
_var329 = ohe_interactions_pipe_1.predict(X_val_0)
_var330 = 1
_var331 = 0.27
_var332 = (_var330 / _var331)
_var333 = (_var329 ** _var332)
r2_score(y_val_0, _var333)
_var334 = 10
_var335 = 10
_var336 = (_var334, _var335)
plt.figure(figsize=_var336)
_var337 = ohe_interactions_pipe_1.predict(X_val_0)
_var338 = 1
_var339 = 0.27
_var340 = (_var338 / _var339)
_var341 = (_var337 ** _var340)
_var342 = 0.1
plt.scatter(y_val_0, _var341, alpha=_var342)
_var343 = plt.gca()
_var344 = 'equal'
_var343.set_aspect(_var344)
_var345 = y_val_0.max()
_var346 = [0, _var345]
_var347 = y_val_0.max()
_var348 = [0, _var347]
_var349 = 'k'
plt.plot(_var346, _var348, c=_var349)
_var350 = np.int
_var351 = X_selected_.astype(_var350)
_var352 = data_nr.columns
_var353 = select_0.get_support()
_var354 = _var352[_var353]
grr = pd.DataFrame(_var351, columns=_var354)
_var355 = grr.columns
_var356 = '$'
df_dummies = pd.get_dummies(grr, columns=_var355, prefix_sep=_var356)
X_dummies_ = df_dummies.values
_var361 = 0
(_var357, _var358, _var359, _var360) = train_test_split(X_dummies_, y_, random_state=_var361)
X_trainval_1 = _var357
X_test_1 = _var358
y_trainval_1 = _var359
y_test_1 = _var360
_var366 = 0
(_var362, _var363, _var364, _var365) = train_test_split(X_trainval_1, y_trainval_1, random_state=_var366)
X_train_1 = _var362
X_val_1 = _var363
y_train_1 = _var364
y_val_1 = _var365
_var367 = (- 3)
_var368 = 2
_var369 = 11
alphas = np.logspace(_var367, _var368, _var369)
alphas
_var370 = True
_var371 = RidgeCV(alphas=alphas, store_cv_values=_var370)
ridge = _var371.fit(X_train_1, y_train_1)
_var371_0 = ridge
ridge.score(X_val_1, y_val_1)
ridge.alpha_
_var372 = ridge.cv_values_
_var373 = 0
_var374 = _var372.mean(axis=_var373)
plt.plot(_var374)
_var375 = 10
_var376 = 20
_var377 = (_var375, _var376)
plt.figure(figsize=_var377)
_var378 = ridge.coef_
inds_0 = np.argsort(_var378)
_var379 = len(inds_0)
_var380 = range(_var379)
_var381 = list(_var380)
_var382 = ridge.coef_
_var383 = _var382[inds_0]
plt.barh(_var381, _var383)
_var384 = len(inds_0)
_var385 = range(_var384)
_var386 = list(_var385)
_var387 = [((feature_mapping[x.split('$')[0]] + ' ') + x.split('$')[1]) for x in df_dummies.columns[inds]]
plt.yticks(_var386, _var387)
_var388 = RidgeCV()
_var389 = (y_train_1 ** l)
ridge_0 = _var388.fit(X_train_1, _var389)
_var388_0 = ridge_0
_var390 = ridge_0.predict(X_val_1)
_var391 = 1
_var392 = (_var391 / l)
_var393 = (_var390 ** _var392)
r2_score(y_val_1, _var393)
_var394 = 10
_var395 = 10
_var396 = (_var394, _var395)
plt.figure(figsize=_var396)
_var397 = ridge_0.predict(X_val_1)
_var398 = 1
_var399 = (_var398 / l)
_var400 = (_var397 ** _var399)
_var401 = 0.1
plt.scatter(y_val_1, _var400, alpha=_var401)
_var402 = plt.gca()
_var403 = 'equal'
_var402.set_aspect(_var403)
_var404 = y_val_1.max()
_var405 = [0, _var404]
_var406 = y_val_1.max()
_var407 = [0, _var406]
_var408 = 'k'
plt.plot(_var405, _var407, c=_var408)
_var409 = RidgeCV()
ridge_1 = _var409.fit(X_train_1, y_train_1)
_var409_0 = ridge_1
_var410 = 10
_var411 = 10
_var412 = (_var410, _var411)
plt.figure(figsize=_var412)
_var413 = ridge_1.predict(X_val_1)
_var414 = 0.1
plt.scatter(y_val_1, _var413, alpha=_var414)
_var415 = plt.gca()
_var416 = 'equal'
_var415.set_aspect(_var416)
_var417 = y_val_1.max()
_var418 = [0, _var417]
_var419 = y_val_1.max()
_var420 = [0, _var419]
_var421 = 'k'
plt.plot(_var418, _var420, c=_var421)
_var422 = 'new_csr'
_var423 = data_[_var422]
_var423.value_counts()
df_dummies.columns
_var424 = 'new_csr$80'
_var425 = df_dummies[_var424]
_var426 = 1
_var427 = (_var425 == _var426)
_var428 = _var427.values
y_[_var428]
_var429 = 'new_csr$80'
_var430 = df_dummies[_var429]
_var431 = 1
_var432 = (_var430 == _var431)
_var433 = df_dummies[_var432]
X_dummies__0 = _var433.values
_var438 = 'new_csr$80'
_var439 = df_dummies[_var438]
_var440 = 1
_var441 = (_var439 == _var440)
_var442 = _var441.values
_var443 = y_[_var442]
_var444 = 0
(_var434, _var435, _var436, _var437) = train_test_split(X_dummies__0, _var443, random_state=_var444)
X_trainval_2 = _var434
X_test_2 = _var435
y_trainval_2 = _var436
y_test_2 = _var437
_var449 = 0
(_var445, _var446, _var447, _var448) = train_test_split(X_trainval_2, y_trainval_2, random_state=_var449)
X_train_2 = _var445
X_val_2 = _var446
y_train_2 = _var447
y_val_2 = _var448
_var450 = RidgeCV()
ridge_2 = _var450.fit(X_train_2, y_train_2)
_var450_0 = ridge_2
_var451 = 10
_var452 = 10
_var453 = (_var451, _var452)
plt.figure(figsize=_var453)
_var454 = ridge_2.predict(X_val_2)
_var455 = 0.1
plt.scatter(y_val_2, _var454, alpha=_var455)
_var456 = plt.gca()
_var457 = 'equal'
_var456.set_aspect(_var457)
_var458 = y_val_2.max()
_var459 = [0, _var458]
_var460 = y_val_2.max()
_var461 = [0, _var460]
_var462 = 'k'
plt.plot(_var459, _var461, c=_var462)
ridge_2.score(X_train_2, y_train_2)
ridge_2.score(X_val_2, y_val_2)
ridge_2.coef_
_var463 = global_wrapper(make)
asdf = _var463
_var464 = True
_var465 = PolynomialFeatures(interaction_only=_var464)
_var466 = VarianceThreshold()
_var467 = 0
_var468 = 4
_var469 = 8
_var470 = np.logspace(_var467, _var468, _var469)
_var471 = RidgeCV(alphas=_var470)
asdf_0 = make_pipeline(_var465, _var466, _var471)
asdf_1 = asdf_0.fit(X_train_2, y_train_2)
_var472 = asdf_1.named_steps
_var473 = 'ridgecv'
_var474 = _var472[_var473]
_var474.alpha_
asdf_1.score(X_train_2, y_train_2)
asdf_1.score(X_val_2, y_val_2)
_var475 = 10
_var476 = 20
_var477 = (_var475, _var476)
plt.figure(figsize=_var477)
_var478 = ridge_2.coef_
inds_1 = np.argsort(_var478)
_var479 = len(inds_1)
_var480 = range(_var479)
_var481 = list(_var480)
_var482 = ridge_2.coef_
_var483 = _var482[inds_1]
plt.barh(_var481, _var483)
_var484 = len(inds_1)
_var485 = range(_var484)
_var486 = list(_var485)
_var487 = [((feature_mapping[x.split('$')[0]] + ' ') + x.split('$')[1]) for x in df_dummies.columns[inds]]
plt.yticks(_var486, _var487)
