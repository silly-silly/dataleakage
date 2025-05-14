

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
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import skew
from scipy.stats.stats import pearsonr
_var0 = get_ipython()
_var1 = 'config'
_var2 = "InlineBackend.figure_format = 'png' #set 'png' here when working on notebook"
_var0.run_line_magic(_var1, _var2)
_var3 = get_ipython()
_var4 = 'matplotlib'
_var5 = 'inline'
_var3.run_line_magic(_var4, _var5)
_var6 = '~/Learning/Data/kaggle/house_prices/train.csv'
train = pd.read_csv(_var6)
_var7 = '~/Learning/Data/kaggle/house_prices/test.csv'
test = pd.read_csv(_var7)
_var8 = train.loc
_var9 = 'MSSubClass'
_var10 = 'SaleCondition'
_var11 = _var8[:, _var9:_var10]
_var12 = test.loc
_var13 = 'MSSubClass'
_var14 = 'SaleCondition'
_var15 = _var12[:, _var13:_var14]
_var16 = (_var11, _var15)
all_data = pd.concat(_var16)
_var17 = matplotlib.rcParams
_var18 = 'figure.figsize'
_var19 = 12.0
_var20 = 6.0
_var21 = (_var19, _var20)
_var17_0 = set_index_wrapper(_var17, _var18, _var21)
_var22 = 'SalePrice'
_var23 = train[_var22]
_var24 = 'SalePrice'
_var25 = train[_var24]
_var26 = np.log1p(_var25)
_var27 = {'price': _var23, 'log(price + 1)': _var26}
prices = pd.DataFrame(_var27)
prices.hist()
_var28 = 'SalePrice'
_var29 = 'SalePrice'
_var30 = train[_var29]
_var31 = np.log1p(_var30)
train_0 = set_index_wrapper(train, _var28, _var31)
_var32 = all_data.dtypes
_var33 = all_data.dtypes
_var34 = 'object'
_var35 = (_var33 != _var34)
_var36 = _var32[_var35]
numeric_feats = _var36.index
numeric_feats
_var37 = train_0[numeric_feats]

def _func0(x):
    _var38 = x.dropna()
    _var39 = skew(_var38)
    return _var39
skewed_feats = _var37.apply(_func0)
_var40 = 0.75
_var41 = (skewed_feats > _var40)
skewed_feats_0 = skewed_feats[_var41]
skewed_feats_1 = skewed_feats_0.index
_var42 = all_data[skewed_feats_1]
_var43 = np.log1p(_var42)
all_data_0 = set_index_wrapper(all_data, skewed_feats_1, _var43)
all_data_1 = pd.get_dummies(all_data_0)
_var44 = all_data_1.mean()
all_data_2 = all_data_1.fillna(_var44)
_var45 = train_0.shape
_var46 = 0
_var47 = _var45[_var46]
X_train = all_data_2[:_var47]
_var48 = train_0.shape
_var49 = 0
_var50 = _var48[_var49]
X_test = all_data_2[_var50:]
y = train_0.SalePrice
from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV
from sklearn.model_selection import cross_val_score

def rmse_cv(model, X_train_0, y_0):
    _var51 = 'neg_mean_squared_error'
    _var52 = 5
    _var53 = cross_val_score(model, X_train_0, y_0, scoring=_var51, cv=_var52)
    _var54 = (- _var53)
    rmse = np.sqrt(_var54)
    return rmse
model_ridge = Ridge()
alphas = [0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 50, 75]
cv_ridge = [rmse_cv(Ridge(alpha=alpha), X_train, y).mean() for alpha in alphas]
cv_ridge_0 = pd.Series(cv_ridge, index=alphas)
_var55 = 'Validation - Just Do It'
cv_ridge_0.plot(title=_var55)
_var56 = 'alpha'
plt.xlabel(_var56)
_var57 = 'rmse'
plt.ylabel(_var57)
cv_ridge_0.min()
_var58 = [1, 0.1, 0.001, 0.0005]
_var59 = LassoCV(alphas=_var58)
model_lasso = _var59.fit(X_train, y)
_var59_0 = model_lasso
_var60 = rmse_cv(model_lasso, X_train, y)
_var60.mean()
_var61 = model_lasso.coef_
_var62 = X_train.columns
coef = pd.Series(_var61, index=_var62)
_var63 = 'Lasso picked '
_var64 = 0
_var65 = (coef != _var64)
_var66 = sum(_var65)
_var67 = str(_var66)
_var68 = (_var63 + _var67)
_var69 = ' variables and eliminated the other '
_var70 = (_var68 + _var69)
_var71 = 0
_var72 = (coef == _var71)
_var73 = sum(_var72)
_var74 = str(_var73)
_var75 = (_var70 + _var74)
_var76 = ' variables'
_var77 = (_var75 + _var76)
print(_var77)
_var78 = coef.sort_values()
_var79 = 5
_var80 = _var78.head(_var79)
_var81 = coef.sort_values()
_var82 = 5
_var83 = _var81.tail(_var82)
_var84 = [_var80, _var83]
imp_coef = pd.concat(_var84)
_var85 = matplotlib.rcParams
_var86 = 'figure.figsize'
_var87 = 8.0
_var88 = 4.0
_var89 = (_var87, _var88)
_var85_0 = set_index_wrapper(_var85, _var86, _var89)
_var90 = 'barh'
imp_coef.plot(kind=_var90)
_var91 = 'Coefficients in the Lasso Model'
plt.title(_var91)
_var92 = matplotlib.rcParams
_var93 = 'figure.figsize'
_var94 = 6.0
_var95 = 4.0
_var96 = (_var94, _var95)
_var92_0 = set_index_wrapper(_var92, _var93, _var96)
_var97 = model_lasso.predict(X_train)
_var98 = {'preds': _var97, 'true': y}
preds = pd.DataFrame(_var98)
_var99 = 'residuals'
_var100 = 'true'
_var101 = preds[_var100]
_var102 = 'preds'
_var103 = preds[_var102]
_var104 = (_var101 - _var103)
preds_0 = set_index_wrapper(preds, _var99, _var104)
_var105 = 'preds'
_var106 = 'residuals'
_var107 = 'scatter'
preds_0.plot(x=_var105, y=_var106, kind=_var107)
import xgboost as xgb
dtrain = xgb.DMatrix(X_train, label=y)
dtest = xgb.DMatrix(X_test)
params = {'max_depth': 10, 'eta': 0.1}
_var108 = 500
_var109 = 100
model_0 = xgb.cv(params, dtrain, num_boost_round=_var108, early_stopping_rounds=_var109)
_var110 = model_0.loc
_var111 = 30
_var112 = ['test-rmse-mean', 'train-rmse-mean']
_var113 = _var110[_var111:, _var112]
_var113.plot()
_var114 = 360
_var115 = 2
_var116 = 0.1
model_xgb = xgb.XGBRegressor(n_estimators=_var114, max_depth=_var115, learning_rate=_var116)
model_xgb_0 = model_xgb.fit(X_train, y)
_var117 = model_xgb_0.predict(X_test)
xgb_preds = np.expm1(_var117)
_var118 = model_lasso.predict(X_test)
lasso_preds = np.expm1(_var118)
_var119 = {'xgb': xgb_preds, 'lasso': lasso_preds}
predictions = pd.DataFrame(_var119)
_var120 = 'xgb'
_var121 = 'lasso'
_var122 = 'scatter'
predictions.plot(x=_var120, y=_var121, kind=_var122)
_var123 = 0.7
_var124 = (_var123 * lasso_preds)
_var125 = 0.3
_var126 = (_var125 * xgb_preds)
preds_1 = (_var124 + _var126)
_var127 = test.Id
_var128 = {'id': _var127, 'SalePrice': preds_1}
solution = pd.DataFrame(_var128)
_var129 = 'ridge_sol.csv'
_var130 = False
solution.to_csv(_var129, index=_var130)
from keras.layers import Dense
from keras.models import Sequential
from keras.regularizers import l1
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
_var131 = StandardScaler()
X_train_1 = _var131.fit_transform(X_train)
_var136 = 3
(_var132, _var133, _var134, _var135) = train_test_split(X_train_1, y, random_state=_var136)
X_tr = _var132
X_val = _var133
y_tr = _var134
y_val = _var135
X_tr.shape
model_1 = Sequential()
_var137 = 1
_var138 = X_train_1.shape
_var139 = 1
_var140 = _var138[_var139]
_var141 = 0.001
_var142 = l1(_var141)
_var143 = Dense(_var137, input_dim=_var140, W_regularizer=_var142)
model_1.add(_var143)
_var144 = 'mse'
_var145 = 'adam'
model_1.compile(loss=_var144, optimizer=_var145)
model_1.summary()
_var146 = (X_val, y_val)
hist = model_1.fit(X_tr, y_tr, validation_data=_var146)
model_2 = hist
_var147 = model_2.predict(X_val)
_var148 = 0
_var149 = _var147[:, _var148]
_var150 = pd.Series(_var149)
_var150.hist()
_var151 = model_2.predict(X_val)
_var152 = 0
_var151[:, _var152]
