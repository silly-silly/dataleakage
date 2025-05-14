

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
_var0 = '../data/ReviewData100k.csv'
normal_data = pd.read_csv(_var0)
_var1 = '../data/data_stop_word_removed.csv'
data_stop_words_removed = pd.read_csv(_var1)
_var2 = '../data/data_POS_tags.csv'
data_pos_tags = pd.read_csv(_var2)
_var3 = ['text', 'stars']
X_train_norm = normal_data[_var3]
_var4 = ['text', 'stars']
X_train_sw = data_stop_words_removed[_var4]
_var5 = ['text', 'stars']
X_train_pos = data_pos_tags[_var5]
_var6 = X_train_norm.text
_var7 = _var6.isnull()
_var8 = True
_var9 = (_var7 != _var8)
X_train_norm_0 = X_train_norm[_var9]
_var10 = X_train_sw.text
_var11 = _var10.isnull()
_var12 = True
_var13 = (_var11 != _var12)
X_train_sw_0 = X_train_sw[_var13]
_var14 = X_train_pos.text
_var15 = _var14.isnull()
_var16 = True
_var17 = (_var15 != _var16)
X_train_pos_0 = X_train_pos[_var17]
from sklearn.feature_extraction.text import TfidfVectorizer
transformer = TfidfVectorizer()
_var18 = 'stars'
y_norm = X_train_norm_0[_var18]
_var19 = 'text'
_var20 = X_train_norm_0[_var19]
X_train_norm_1 = transformer.fit_transform(_var20)
_var21 = 'stars'
y_sw = X_train_sw_0[_var21]
_var22 = 'text'
_var23 = X_train_sw_0[_var22]
X_train_sw_1 = transformer.fit_transform(_var23)
_var24 = 'stars'
y_pos = X_train_pos_0[_var24]
_var25 = 'text'
_var26 = X_train_pos_0[_var25]
X_train_pos_1 = transformer.fit_transform(_var26)

def rmse(pred, labels):
    _var27 = (pred - labels)
    _var28 = 2
    _var29 = (_var27 ** _var28)
    _var30 = np.mean(_var29)
    _var31 = np.sqrt(_var30)
    return _var31
RANDOM_STATE = 2016
from sklearn.model_selection import train_test_split
_var36 = 0.33
(_var32, _var33, _var34, _var35) = train_test_split(X_train_pos_1, y_pos, test_size=_var36, random_state=RANDOM_STATE)
X_train = _var32
X_test = _var33
y_train = _var34
y_test = _var35
_var37 = np.mean(y_train)
_var38 = [_var37]
_var39 = len(y_test)
y_pred_base = (_var38 * _var39)
_var40 = 'Baseline model accuracy: '
_var41 = rmse(y_pred_base, y_test)
_var42 = (_var40, _var41)
print(_var42)
from sklearn.linear_model import LinearRegression
lr_model = LinearRegression()
lr_model_0 = lr_model.fit(X_train, y_train)
y_pred_lr = lr_model_0.predict(X_test)
_var43 = 'Ridge model accuracy: '
_var44 = rmse(y_pred_lr, y_test)
_var45 = (_var43, _var44)
print(_var45)
from sklearn.linear_model import RidgeCV, LassoCV
_var46 = [0.01, 0.05, 0.1, 0.2, 0.5, 1]
ridge_model = RidgeCV(alphas=_var46)
_var47 = [0.01, 0.05, 0.1, 0.2, 0.5, 1]
lasso_model = LassoCV(alphas=_var47)
ridge_model_0 = ridge_model.fit(X_train, y_train)
lasso_model_0 = lasso_model.fit(X_train, y_train)
y_pred_rm = ridge_model_0.predict(X_test)
y_pred_lm = lasso_model_0.predict(X_test)
_var48 = 'Ridge model accuracy: '
_var49 = rmse(y_pred_rm, y_test)
_var50 = (_var48, _var49)
print(_var50)
_var51 = 'Lasso model accuracy: '
_var52 = rmse(y_pred_lm, y_test)
_var53 = (_var51, _var52)
print(_var53)
from sklearn.neighbors import KNeighborsRegressor
_var54 = 5
knn_model5 = KNeighborsRegressor(n_neighbors=_var54)
_var55 = 10
knn_model10 = KNeighborsRegressor(n_neighbors=_var55)
_var56 = 50
knn_model50 = KNeighborsRegressor(n_neighbors=_var56)
knn_model5_0 = knn_model5.fit(X_train, y_train)
knn_model10_0 = knn_model10.fit(X_train, y_train)
knn_model50_0 = knn_model50.fit(X_train, y_train)
y_pred_knn5 = knn_model5_0.predict(X_test)
y_pred_knn10 = knn_model10_0.predict(X_test)
y_pred_knn50 = knn_model50_0.predict(X_test)
_var57 = 'knn model with 5 neighbours accuracy: '
_var58 = rmse(y_pred_knn5, y_test)
_var59 = (_var57, _var58)
print(_var59)
_var60 = 'knn model with 10 neighbours accuracy: '
_var61 = rmse(y_pred_knn10, y_test)
_var62 = (_var60, _var61)
print(_var62)
_var63 = 'knn model with 50 neighbours accuracy: '
_var64 = rmse(y_pred_knn50, y_test)
_var65 = (_var63, _var64)
print(_var65)
from sklearn.tree import DecisionTreeRegressor
dt_model = DecisionTreeRegressor(random_state=RANDOM_STATE)
dt_model_0 = dt_model.fit(X_train, y_train)
y_pred_dt = dt_model_0.predict(X_test)
_var66 = 'Decision tree accuracy: '
_var67 = rmse(y_pred_dt, y_test)
_var68 = (_var66, _var67)
print(_var68)
from sklearn.ensemble import RandomForestRegressor
_var69 = 4
_var70 = 100
_var71 = 'sqrt'
_var72 = 1
rf_model = RandomForestRegressor(max_depth=_var69, n_estimators=_var70, max_features=_var71, verbose=_var72, random_state=RANDOM_STATE)
rf_model_0 = rf_model.fit(X_train, y_train)
y_pred_rf = rf_model_0.predict(X_test)
_var73 = 'Randomforest with 100 estimators accuracy: '
_var74 = rmse(y_pred_rf, y_test)
_var75 = (_var73, _var74)
print(_var75)
from sklearn.ensemble import AdaBoostRegressor
_var76 = 100
_var77 = 0.01
_var78 = 'square'
adb_model = AdaBoostRegressor(n_estimators=_var76, learning_rate=_var77, random_state=RANDOM_STATE, loss=_var78)
adb_model_0 = adb_model.fit(X_train, y_train)
y_pred_adb = adb_model_0.predict(X_test)
_var79 = 'Adaboost with 100 estimators accuracy: '
_var80 = rmse(y_pred_adb, y_test)
_var81 = (_var79, _var80)
print(_var81)
from sklearn.ensemble import GradientBoostingRegressor
_var82 = 100
_var83 = 0.01
_var84 = 4
_var85 = 'sqrt'
gbm_model = GradientBoostingRegressor(n_estimators=_var82, learning_rate=_var83, random_state=RANDOM_STATE, max_depth=_var84, max_features=_var85)
gbm_model_0 = gbm_model.fit(X_train, y_train)
_var86 = X_test.todense()
y_pred_gbm = gbm_model_0.predict(_var86)
_var87 = 'GBM with 100 estimators accuracy: '
_var88 = rmse(y_pred_gbm, y_test)
_var89 = (_var87, _var88)
print(_var89)
