

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
import sys
from urllib.request import urlopen
import numpy as np
import pandas as pd
from pandas.tools.plotting import scatter_matrix
import matplotlib
import matplotlib.pyplot as plt
_var0 = get_ipython()
_var1 = 'matplotlib'
_var2 = 'inline'
_var0.run_line_magic(_var1, _var2)
from math import isnan
import seaborn as sns
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
url = 'https://s3.amazonaws.com/nyc-tlc/trip+data/green_tripdata_2015-09.csv'
_var3 = urlopen(url)
df = pd.read_csv(_var3)
_var4 = 'n_rows = %d, n_columns = %d'
_var5 = df.shape
_var6 = (_var4 % _var5)
print(_var6)
_var7 = 2
df.head(_var7)
_var8 = 8
_var9 = 4
_var10 = (_var8, _var9)
fig = plt.figure(figsize=_var10)
_var11 = 1
_var12 = 2
_var13 = 1
ax1 = fig.add_subplot(_var11, _var12, _var13)
_var14 = 1
_var15 = 2
_var16 = 2
ax2 = fig.add_subplot(_var14, _var15, _var16)
_var17 = 'Trip_distance'
_var18 = df[_var17]
_var19 = _var18.values
_var20 = 500
ax1.hist(_var19, bins=_var20)
_var21 = 'Number of Trips'
ax1.set_ylabel(_var21)
_var22 = 'Miles'
ax1.set_xlabel(_var22)
_var23 = [0, 50]
ax1.set_xlim(_var23)
_var24 = 'Trip_distance'
_var25 = df[_var24]
_var26 = _var25.values
ax2.boxplot(_var26)
_var27 = 'log'
ax2.set_yscale(_var27)
_var28 = 'Miles'
ax2.set_ylabel(_var28)
_var29 = ax2.get_xaxis()
_var30 = []
_var29.set_ticks(_var30)
_var31 = '2 Miles'
_var32 = 1
_var33 = 2
_var34 = (_var32, _var33)
_var35 = 1.1
_var36 = 2
_var37 = (_var35, _var36)
ax2.annotate(_var31, xy=_var34, xytext=_var37)
fig.tight_layout()
_var38 = 'Pickup_hour'
_var39 = df.lpep_pickup_datetime
_var40 = pd.to_datetime(_var39)
_var41 = df.set_index(_var40)
_var42 = _var41.index
_var43 = _var42.hour
df_0 = set_index_wrapper(df, _var38, _var43)
_var44 = 'Dropoff_hour'
_var45 = df_0.Lpep_dropoff_datetime
_var46 = pd.to_datetime(_var45)
_var47 = df_0.set_index(_var46)
_var48 = _var47.index
_var49 = _var48.hour
df_1 = set_index_wrapper(df_0, _var44, _var49)
_var50 = 'Pickup_hour'
_var51 = 'Trip_distance'
_var52 = 'mean'
_var53 = df_1.pivot_table(index=_var50, values=_var51, aggfunc=_var52)
_var54 = 2
Table1 = _var53.round(_var54)

def plot_pickup():
    _var55 = global_wrapper(Table1)
    p = _var55.plot()
    _var56 = 'Miles'
    p.set_ylabel(_var56)
    _var57 = 'Hours after midnight'
    p.set_xlabel(_var57)
    _var58 = 'Trip distance by Pickup_hour'
    p.set_title(_var58)
plot_pickup()

def twoAirports():
    _var59 = global_wrapper(df_1)
    _var60 = global_wrapper(df_1)
    _var61 = _var60.RateCodeID
    _var62 = 2
    _var63 = (_var61 == _var62)
    _var64 = _var59[_var63]
    _var65 = _var64.shape
    _var66 = 0
    jfk = _var65[_var66]
    _var67 = global_wrapper(df_1)
    _var68 = global_wrapper(df_1)
    _var69 = _var68.RateCodeID
    _var70 = 3
    _var71 = (_var69 == _var70)
    _var72 = _var67[_var71]
    _var73 = _var72.shape
    _var74 = 0
    newark = _var73[_var74]
    _var75 = global_wrapper(df_1)
    _var76 = _var75.Fare_amount
    _var77 = global_wrapper(df_1)
    _var78 = _var77.RateCodeID
    _var79 = [_var78]
    _var80 = _var76.groupby(_var79)
    avgf = _var80.mean()
    _var81 = global_wrapper(df_1)
    _var82 = _var81.Total_amount
    _var83 = global_wrapper(df_1)
    _var84 = _var83.RateCodeID
    _var85 = [_var84]
    _var86 = _var82.groupby(_var85)
    avgt = _var86.mean()
    _var87 = (jfk, newark)
    _var88 = 2
    _var89 = avgf[_var88]
    _var90 = 3
    _var91 = avgf[_var90]
    _var92 = (_var89, _var91)
    _var93 = 2
    _var94 = np.round(_var92, _var93)
    _var95 = 2
    _var96 = avgt[_var95]
    _var97 = 3
    _var98 = avgt[_var97]
    _var99 = (_var96, _var98)
    _var100 = 2
    _var101 = np.round(_var99, _var100)
    _var102 = 2
    _var103 = 3
    _var104 = (_var102, _var103)
    _var105 = {'Trips': _var87, 'average Fare': _var94, 'average Total': _var101, 'RateCode': _var104}
    _var106 = ['JFK', 'Newark']
    airports = pd.DataFrame(_var105, index=_var106)
    print(airports)
    _var107 = ['average Fare', 'average Total']
    _var108 = airports[_var107]
    _var109 = _var108.plot
    _var109.barh()
twoAirports()

def preproc(data):
    'Return a copy of clean data with Tip_percent.'
    _var110 = data.Payment_type
    _var111 = 1
    _var112 = (_var110 == _var111)
    data_0 = data[_var112]
    _var113 = ['Payment_type', 'lpep_pickup_datetime', 'Dropoff_latitude', 'Lpep_dropoff_datetime', 'Ehail_fee', 'Pickup_latitude', 'Pickup_longitude', 'Dropoff_longitude']
    _var114 = 1
    data_1 = data_0.drop(_var113, axis=_var114)
    _var115 = 'ffill'
    _var116 = True
    data_1.fillna(method=_var115, inplace=_var116)
    _var117 = data_1.Store_and_fwd_flag
    _var118 = 'Y'
    _var119 = (_var117 == _var118)
    _var120 = _var119.astype(int)
    data_2 = set_field_wrapper(data_1, 'Store_and_fwd_flag', _var120)
    _var121 = 'Trip_type '
    _var122 = 'Trip_type '
    _var123 = data_2[_var122]
    _var124 = 'int'
    _var125 = _var123.astype(_var124)
    data_3 = set_index_wrapper(data_2, _var121, _var125)
    fields = ['Fare_amount', 'Extra', 'MTA_tax', 'Tip_amount', 'Total_amount']
    for field in fields:
        _var126 = data_3[field]
        _var127 = _var126.abs()
        data_4 = set_index_wrapper(data_3, field, _var127)
    data_5 = __phi__(data_4, data_3)
    _var128 = 'Tip_percent'
    _var129 = data_5.Tip_amount
    _var130 = data_5.Total_amount
    _var131 = (_var129 / _var130)
    data_6 = set_index_wrapper(data_5, _var128, _var131)
    _var132 = data_6.Tip_percent
    _var133 = data_6.Tip_percent
    _var134 = _var133.mean()
    _var135 = _var132.fillna(_var134)
    data_7 = set_field_wrapper(data_6, 'Tip_percent', _var135)
    _var136 = 'Tip_amount'
    _var137 = 1
    _var138 = True
    data_7.drop(_var136, axis=_var137, inplace=_var138)
    return data_7
data_8 = preproc(df_1)
_var139 = 16
_var140 = 12
_var141 = (_var139, _var140)
data_8.hist(figsize=_var141)

def find_corr_features():
    'correlation matrix'
    _var142 = global_wrapper(data_8)
    _var143 = 0.005
    dsmall = _var142.sample(frac=_var143)
    corr = dsmall.corr()
    _var144 = 0.7
    _var145 = (corr > _var144)
    _var146 = 0.99
    _var147 = (corr < _var146)
    _var148 = (_var145 & _var147)
    df_2 = corr[_var148]
    _var149 = 'white'
    sns.set_style(_var149)
    _var150 = np.bool
    _var151 = np.zeros_like(df_2, dtype=_var150)
    _var152 = True
    _var153 = 220
    _var154 = 10
    _var155 = True
    _var156 = sns.diverging_palette(_var153, _var154, as_cmap=_var155)
    sns.heatmap(df_2, mask=_var151, square=_var152, cmap=_var156)
    pairs = set()
    return df_2
tmp = find_corr_features()

def redundant_features(corr_0, data_9):
    'drop redundant features based on corr'
    d = corr_0.to_dict()
    redundant_columns = []
    pairs_0 = []
    for k1 in d:
        _var157 = d[k1]
        for k2 in _var157:
            _var158 = d[k1]
            _var159 = _var158[k2]
            _var160 = isnan(_var159)
            _var161 = (not _var160)
            if _var161:
                _var162 = (k1, k2)
                _var163 = (_var162 not in pairs_0)
                _var164 = (k2, k1)
                _var165 = (_var164 not in pairs_0)
                _var166 = (_var163 & _var165)
                if _var166:
                    _var167 = (k1, k2)
                    pairs_0.append(_var167)
                    _var168 = '\t'
                    _var169 = '\t'
                    _var170 = '%.2f'
                    _var171 = d[k1]
                    _var172 = _var171[k2]
                    _var173 = (_var170 % _var172)
                    _var174 = (k1, _var168, k2, _var169, _var173)
                    print(_var174)
                    redundant_columns.append(k2)
    _var175 = 1
    _var176 = True
    data_9.drop(redundant_columns, axis=_var175, inplace=_var176)
    return data_9
data_10 = redundant_features(tmp, data_8)

def rfm(data_11):
    target = 'Tip_percent'
    features = [i for i in data.columns if (i != target)]
    _var181 = data_11[features]
    _var182 = data_11[target]
    _var183 = 0.3
    _var184 = 0
    (_var177, _var178, _var179, _var180) = train_test_split(_var181, _var182, test_size=_var183, random_state=_var184)
    X_train = _var177
    X_test = _var178
    y_train = _var179
    y_test = _var180
    _var185 = 10
    _var186 = 2
    rf = RandomForestRegressor(n_estimators=_var185, n_jobs=_var186)
    rf_0 = rf.fit(X_train, y_train)
    print(rf_0)
    y_pred = rf_0.predict(X_test)
    _var187 = rf_0.predict(X_test)
    mae = mean_absolute_error(y_test, _var187)
    _var188 = r2_score(y_test, y_pred)
    _var189 = '0.4f'
    rsqr = format(_var188, _var189)
    _var190 = 'Mean absolute error:'
    _var191 = '\tR2:'
    _var192 = (_var190, mae, _var191, rsqr)
    print(_var192)
    return (y_pred, y_test, X_test, rf_0)
(_var193, _var194, _var195, _var196) = rfm(data_10)
y_pred_0 = _var193
y_test_0 = _var194
X_test_0 = _var195
rf_1 = _var196

def plot_rf():
    _var197 = 8
    _var198 = 4
    _var199 = (_var197, _var198)
    fig_0 = plt.figure(figsize=_var199)
    _var200 = 1
    _var201 = 2
    _var202 = 1
    ax1_0 = fig_0.add_subplot(_var200, _var201, _var202)
    _var203 = 1
    _var204 = 2
    _var205 = 2
    ax2_0 = fig_0.add_subplot(_var203, _var204, _var205)
    _var206 = global_wrapper(y_test_0)
    _var207 = global_wrapper(y_pred_0)
    ax1_0.scatter(_var206, _var207)
    _var208 = 'y_test'
    ax1_0.set_xlabel(_var208)
    _var209 = 'y_predicted'
    ax1_0.set_ylabel(_var209)
    _var210 = global_wrapper(X_test_0)
    _var211 = _var210.columns
    _var212 = global_wrapper(rf_1)
    _var213 = _var212.feature_importances_
    _var214 = zip(_var211, _var213)

    def _func0(x):
        _var215 = 1
        _var216 = x[_var215]
        return _var216
    _var217 = True
    importance_data = sorted(_var214, key=_func0, reverse=_var217)
    _var220 = zip(*importance_data)
    (_var218, _var219) = list(_var220)
    xlab = _var218
    ylab = _var219
    _var221 = global_wrapper(X_test_0)
    _var222 = _var221.columns
    _var223 = len(_var222)
    _var224 = range(_var223)
    xloc = list(_var224)
    ax2_0.barh(xloc, ylab)
    ax2_0.set_yticks(xloc)
    ax2_0.set_yticklabels(xlab)
    _var225 = 'Random Forest Feature Importance'
    ax2_0.set_title(_var225)
    fig_0.tight_layout()
plot_rf()

def get_speed_week(dat):
    _var226 = dat.Lpep_dropoff_datetime
    end = pd.to_datetime(_var226)
    _var227 = dat.lpep_pickup_datetime
    begin = pd.to_datetime(_var227)
    _var228 = (end - begin)
    _var229 = 'timedelta64[s]'
    duration = _var228.astype(_var229)
    _var230 = dat.Trip_distance
    _var231 = (_var230 / duration)
    _var232 = 3600
    speed = (_var231 * _var232)
    _var233 = dat.lpep_pickup_datetime
    _var234 = pd.to_datetime(_var233)
    _var235 = dat.set_index(_var234)
    _var236 = _var235.index
    week = _var236.week
    return (speed, week)
(_var237, _var238) = get_speed_week(data_10)
_var239 = 'Speed'
data_12 = set_index_wrapper(data_10, _var239, _var237)
_var240 = 'Week'
data_13 = set_index_wrapper(data_12, _var240, _var238)
_var241 = ['Speed', 'Week', 'Pickup_hour']
_var242 = df_1[_var241]
_var243 = np.inf
_var244 = np.inf
_var245 = (- _var244)
_var246 = [_var243, _var245]
_var247 = np.nan
df1 = _var242.replace(_var246, _var247)
_var248 = True
df1.dropna(inplace=_var248)
_var249 = df1.Speed
_var250 = 60
_var251 = (_var249 < _var250)
_var252 = df1[_var251]
_var253 = 'Speed'
_var254 = 'Week'
_var252.boxplot(_var253, by=_var254)
import statsmodels.api as sm
from statsmodels.formula.api import ols
_var255 = df1.Week
_var256 = 'category'
_var257 = _var255.astype(_var256)
df1_0 = set_field_wrapper(df1, 'Week', _var257)
_var258 = 'Speed ~ Week'
_var259 = ols(_var258, data=df1_0)
mod = _var259.fit()
_var259_0 = mod
_var260 = sm.stats
_var261 = 2
aov_table = _var260.anova_lm(mod, typ=_var261)
print(aov_table)
_var262 = 'Speed'
_var263 = 'Pickup_hour'
_var264 = df1_0.pivot_table(_var262, _var263)
plt.plot(_var264)
_var265 = 'Average speed by time of day'
plt.title(_var265)
_var266 = 'Hours after midnight'
plt.xlabel(_var266)
_var267 = 'mils per hour'
plt.ylabel(_var267)
_var268 = df1_0.Pickup_hour
_var269 = 'category'
_var270 = _var268.astype(_var269)
df1_1 = set_field_wrapper(df1_0, 'Pickup_hour', _var270)
_var271 = 'Speed ~ Pickup_hour'
_var272 = ols(_var271, data=df1_1)
mod_0 = _var272.fit()
_var272_0 = mod_0
_var273 = sm.stats
_var274 = 2
aov_table_0 = _var273.anova_lm(mod_0, typ=_var274)
print(aov_table_0)
