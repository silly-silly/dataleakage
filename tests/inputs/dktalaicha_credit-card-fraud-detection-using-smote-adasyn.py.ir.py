

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
import warnings
_var0 = 'ignore'
warnings.filterwarnings(_var0)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly
import datetime
import math
import matplotlib
import sklearn
from IPython.display import HTML
from IPython.display import YouTubeVideo
import pickle
import os
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
_var1 = np.__version__
_var2 = f'Numpy version : Numpy {_var1}'
print(_var2)
_var3 = pd.__version__
_var4 = f'Pandas version : Pandas {_var3}'
print(_var4)
_var5 = matplotlib.__version__
_var6 = f'Matplotlib version : Matplotlib {_var5}'
print(_var6)
_var7 = sns.__version__
_var8 = f'Seaborn version : Seaborn {_var7}'
print(_var8)
_var9 = sklearn.__version__
_var10 = f'SkLearn version : SkLearn {_var9}'
print(_var10)
_var11 = plotly.__version__
_var12 = f'Plotly version : plotly {_var11}'
print(_var12)
_var13 = get_ipython()
_var14 = 'matplotlib'
_var15 = 'inline'
_var13.run_line_magic(_var14, _var15)
_var16 = 'darkgrid'
_var17 = 'colorblind'
sns.set(style=_var16, palette=_var17)
_var18 = '../input/creditcardfraud/creditcard.csv'
_var19 = 'latin_1'
df = pd.read_csv(_var18, encoding=_var19)
_var20 = df.columns
_var21 = _var20.str
_var22 = _var21.lower()
df_0 = set_field_wrapper(df, 'columns', _var22)
df_0.head()
df_0.tail()
_var23 = pd.options
_var24 = _var23.display
_var25 = 100
_var24_0 = set_field_wrapper(_var24, 'max_rows', _var25)
_var26 = pd.options
_var27 = _var26.display
_var28 = 100
_var27_0 = set_field_wrapper(_var27, 'max_columns', _var28)
_var29 = 10
df_0.head(_var29)
df_0.info()
_var30 = 'class'
_var31 = df_0[_var30]
_var32 = _var31.value_counts()
print(_var32)
_var33 = '\n'
print(_var33)
_var34 = 'class'
_var35 = df_0[_var34]
_var36 = True
_var37 = _var35.value_counts(normalize=_var36)
print(_var37)
_var38 = 'class'
_var39 = df_0[_var38]
_var40 = _var39.value_counts()
_var41 = 'pie'
_var42 = [0, 0.1]
_var43 = 6
_var44 = 6
_var45 = (_var43, _var44)
_var46 = '%1.1f%%'
_var47 = True
_var40.plot(kind=_var41, explode=_var42, figsize=_var45, autopct=_var46, shadow=_var47)
_var48 = 'Fraudulent and Non-Fraudulent Distribution'
_var49 = 20
plt.title(_var48, fontsize=_var49)
_var50 = ['Fraud', 'Genuine']
plt.legend(_var50)
plt.show()
_var51 = ['time', 'amount']
_var52 = df_0[_var51]
_var52.describe()
_var53 = df_0.isnull()
_var54 = _var53.sum()
_var54.max()
_var55 = 8
_var56 = 6
_var57 = (_var55, _var56)
plt.figure(figsize=_var57)
_var58 = 'Distribution of Transaction Amount'
_var59 = 14
plt.title(_var58, fontsize=_var59)
_var60 = 'amount'
_var61 = df_0[_var60]
_var62 = 100
sns.distplot(_var61, bins=_var62)
plt.show()
_var65 = 2
_var66 = 16
_var67 = 4
_var68 = (_var66, _var67)
(_var63, _var64) = plt.subplots(ncols=_var65, figsize=_var68)
fig = _var63
axs = _var64
_var69 = 'class'
_var70 = df_0[_var69]
_var71 = 1
_var72 = (_var70 == _var71)
_var73 = df_0[_var72]
_var74 = 'amount'
_var75 = _var73[_var74]
_var76 = 100
_var77 = 0
_var78 = axs[_var77]
sns.distplot(_var75, bins=_var76, ax=_var78)
_var79 = 0
_var80 = axs[_var79]
_var81 = 'Distribution of Fraud Transactions'
_var80.set_title(_var81)
_var82 = 'class'
_var83 = df_0[_var82]
_var84 = 0
_var85 = (_var83 == _var84)
_var86 = df_0[_var85]
_var87 = 'amount'
_var88 = _var86[_var87]
_var89 = 100
_var90 = 0
_var91 = axs[_var90]
sns.distplot(_var88, bins=_var89, ax=_var91)
_var92 = 1
_var93 = axs[_var92]
_var94 = 'Distribution of Genuine Transactions'
_var93.set_title(_var94)
plt.show()
_var95 = 'Fraud Transaction distribution : \n'
_var96 = 'class'
_var97 = df_0[_var96]
_var98 = 1
_var99 = (_var97 == _var98)
_var100 = df_0[_var99]
_var101 = 'amount'
_var102 = _var100[_var101]
_var103 = _var102.value_counts()
_var104 = _var103.head()
_var105 = (_var95, _var104)
print(_var105)
_var106 = '\n'
print(_var106)
_var107 = 'Maximum amount of fraud transaction - '
_var108 = 'class'
_var109 = df_0[_var108]
_var110 = 1
_var111 = (_var109 == _var110)
_var112 = df_0[_var111]
_var113 = 'amount'
_var114 = _var112[_var113]
_var115 = _var114.max()
_var116 = (_var107, _var115)
print(_var116)
_var117 = 'Minimum amount of fraud transaction - '
_var118 = 'class'
_var119 = df_0[_var118]
_var120 = 1
_var121 = (_var119 == _var120)
_var122 = df_0[_var121]
_var123 = 'amount'
_var124 = _var122[_var123]
_var125 = _var124.min()
_var126 = (_var117, _var125)
print(_var126)
_var127 = 'Genuine Transaction distribution : \n'
_var128 = 'class'
_var129 = df_0[_var128]
_var130 = 0
_var131 = (_var129 == _var130)
_var132 = df_0[_var131]
_var133 = 'amount'
_var134 = _var132[_var133]
_var135 = _var134.value_counts()
_var136 = _var135.head()
_var137 = (_var127, _var136)
print(_var137)
_var138 = '\n'
print(_var138)
_var139 = 'Maximum amount of Genuine transaction - '
_var140 = 'class'
_var141 = df_0[_var140]
_var142 = 0
_var143 = (_var141 == _var142)
_var144 = df_0[_var143]
_var145 = 'amount'
_var146 = _var144[_var145]
_var147 = _var146.max()
_var148 = (_var139, _var147)
print(_var148)
_var149 = 'Minimum amount of Genuine transaction - '
_var150 = 'class'
_var151 = df_0[_var150]
_var152 = 0
_var153 = (_var151 == _var152)
_var154 = df_0[_var153]
_var155 = 'amount'
_var156 = _var154[_var155]
_var157 = _var156.min()
_var158 = (_var149, _var157)
print(_var158)
_var159 = 8
_var160 = 6
_var161 = (_var159, _var160)
plt.figure(figsize=_var161)
_var162 = 'class'
_var163 = 'amount'
sns.boxplot(x=_var162, y=_var163, data=df_0)
_var164 = 'Amount Distribution for Fraud and Genuine transactions'
plt.title(_var164)
plt.show()
_var165 = 8
_var166 = 6
_var167 = (_var165, _var166)
plt.figure(figsize=_var167)
_var168 = 'Distribution of Transaction Time'
_var169 = 14
plt.title(_var168, fontsize=_var169)
_var170 = 'time'
_var171 = df_0[_var170]
_var172 = 100
sns.distplot(_var171, bins=_var172)
plt.show()
_var175 = 2
_var176 = 16
_var177 = 4
_var178 = (_var176, _var177)
(_var173, _var174) = plt.subplots(ncols=_var175, figsize=_var178)
fig_0 = _var173
axs_0 = _var174
_var179 = 'class'
_var180 = df_0[_var179]
_var181 = 1
_var182 = (_var180 == _var181)
_var183 = df_0[_var182]
_var184 = 'time'
_var185 = _var183[_var184]
_var186 = 100
_var187 = 'red'
_var188 = 0
_var189 = axs_0[_var188]
sns.distplot(_var185, bins=_var186, color=_var187, ax=_var189)
_var190 = 0
_var191 = axs_0[_var190]
_var192 = 'Distribution of Fraud Transactions'
_var191.set_title(_var192)
_var193 = 'class'
_var194 = df_0[_var193]
_var195 = 0
_var196 = (_var194 == _var195)
_var197 = df_0[_var196]
_var198 = 'time'
_var199 = _var197[_var198]
_var200 = 100
_var201 = 'green'
_var202 = 1
_var203 = axs_0[_var202]
sns.distplot(_var199, bins=_var200, color=_var201, ax=_var203)
_var204 = 1
_var205 = axs_0[_var204]
_var206 = 'Distribution of Genuine Transactions'
_var205.set_title(_var206)
plt.show()
_var207 = 12
_var208 = 8
_var209 = (_var207, _var208)
plt.figure(figsize=_var209)
_var210 = 'class'
_var211 = 'time'
ax = sns.boxplot(x=_var210, y=_var211, data=df_0)
_var212 = ax.artists
_var213 = 0
_var214 = _var212[_var213]
_var215 = '#90EE90'
_var214.set_facecolor(_var215)
_var216 = ax.artists
_var217 = 1
_var218 = _var216[_var217]
_var219 = '#FA8072'
_var218.set_facecolor(_var219)
_var220 = 'Time Distribution for Fraud and Genuine transactions'
plt.title(_var220)
plt.show()
_var223 = 2
_var224 = True
_var225 = 16
_var226 = 6
_var227 = (_var225, _var226)
(_var221, _var222) = plt.subplots(nrows=_var223, sharex=_var224, figsize=_var227)
fig_1 = _var221
axs_1 = _var222
_var228 = 'time'
_var229 = 'amount'
_var230 = 'class'
_var231 = df_0[_var230]
_var232 = 1
_var233 = (_var231 == _var232)
_var234 = df_0[_var233]
_var235 = 0
_var236 = axs_1[_var235]
sns.scatterplot(x=_var228, y=_var229, data=_var234, ax=_var236)
_var237 = 0
_var238 = axs_1[_var237]
_var239 = 'Distribution of Fraud Transactions'
_var238.set_title(_var239)
_var240 = 'time'
_var241 = 'amount'
_var242 = 'class'
_var243 = df_0[_var242]
_var244 = 0
_var245 = (_var243 == _var244)
_var246 = df_0[_var245]
_var247 = 1
_var248 = axs_1[_var247]
sns.scatterplot(x=_var240, y=_var241, data=_var246, ax=_var248)
_var249 = 1
_var250 = axs_1[_var249]
_var251 = 'Distribution of Genue Transactions'
_var250.set_title(_var251)
plt.show()
_var252 = ['time', 'amount', 'class']
_var253 = df_0[_var252]
_var253.nunique()
_var254 = 'time'
_var255 = 'amount'
_var256 = 'class'
_var257 = 'violin'
_var258 = 'box'
_var259 = 'ols'
_var260 = 'simple_white'
fig_2 = px.scatter(df_0, x=_var254, y=_var255, color=_var256, marginal_y=_var257, marginal_x=_var258, trendline=_var259, template=_var260)
fig_2.show()
_var261 = ['time', 'amount', 'class']
_var262 = df_0[_var261]
_var263 = _var262.corr()
_var264 = 'class'
_var265 = _var263[_var264]
_var266 = False
_var267 = _var265.sort_values(ascending=_var266)
_var268 = 10
_var267.head(_var268)
_var269 = 'Pearson Correlation Matrix'
plt.title(_var269)
_var270 = ['time', 'amount', 'class']
_var271 = df_0[_var270]
_var272 = _var271.corr()
_var273 = 0.25
_var274 = 0.7
_var275 = True
_var276 = 'winter'
_var277 = 'w'
_var278 = True
sns.heatmap(_var272, linewidths=_var273, vmax=_var274, square=_var275, cmap=_var276, linecolor=_var277, annot=_var278)
df_0.shape
_var279 = 'class'
_var280 = df_0[_var279]
_var281 = True
_var280.value_counts(normalize=_var281)
_var282 = 'time'
_var283 = 'time'
_var284 = df_0[_var283]

def _func0(sec):
    _var285 = 3600
    _var286 = (sec / _var285)
    return _var286
_var287 = _var284.apply(_func0)
df_1 = set_index_wrapper(df_0, _var282, _var287)
_var288 = 'hour'
_var289 = 'time'
_var290 = df_1[_var289]
_var291 = 24
_var292 = (_var290 % _var291)
df_2 = set_index_wrapper(df_1, _var288, _var292)
_var293 = 'hour'
_var294 = 'hour'
_var295 = df_2[_var294]

def _func1(x):
    _var296 = math.floor(x)
    return _var296
_var297 = _var295.apply(_func1)
df_3 = set_index_wrapper(df_2, _var293, _var297)
_var298 = 'day'
_var299 = 'time'
_var300 = df_3[_var299]
_var301 = 24
_var302 = (_var300 / _var301)
df_4 = set_index_wrapper(df_3, _var298, _var302)
_var303 = 'day'
_var304 = 'day'
_var305 = df_4[_var304]

def _func2(x_0):
    _var306 = 0
    _var307 = (x_0 == _var306)
    _var308 = 1
    _var309 = math.ceil(x_0)
    _var310 = (_var308 if _var307 else _var309)
    return _var310
_var311 = _var305.apply(_func2)
df_5 = set_index_wrapper(df_4, _var303, _var311)
_var312 = ['time', 'hour', 'day', 'amount', 'class']
df_5[_var312]
_var313 = 'class'
_var314 = df_5[_var313]
_var315 = 1
_var316 = (_var314 == _var315)
_var317 = df_5[_var316]
_var318 = 'day'
_var319 = _var317[_var318]
dayFrdTran = _var319.value_counts()
_var320 = 'class'
_var321 = df_5[_var320]
_var322 = 0
_var323 = (_var321 == _var322)
_var324 = df_5[_var323]
_var325 = 'day'
_var326 = _var324[_var325]
dayGenuTran = _var326.value_counts()
_var327 = 'day'
_var328 = df_5[_var327]
dayTran = _var328.value_counts()
_var329 = 'No of transaction Day wise:'
print(_var329)
print(dayTran)
_var330 = '\n'
print(_var330)
_var331 = 'No of fraud transaction Day wise:'
print(_var331)
print(dayFrdTran)
_var332 = '\n'
print(_var332)
_var333 = 'No of genuine transactions Day wise:'
print(_var333)
print(dayGenuTran)
_var334 = '\n'
print(_var334)
_var335 = 'Percentage of fraud transactions Day wise:'
print(_var335)
_var336 = (dayFrdTran / dayTran)
_var337 = 100
_var338 = (_var336 * _var337)
print(_var338)
_var341 = 3
_var342 = 16
_var343 = 4
_var344 = (_var342, _var343)
(_var339, _var340) = plt.subplots(ncols=_var341, figsize=_var344)
fig_3 = _var339
axs_2 = _var340
_var345 = 'day'
_var346 = df_5[_var345]
_var347 = 0
_var348 = axs_2[_var347]
sns.countplot(_var346, ax=_var348)
_var349 = 0
_var350 = axs_2[_var349]
_var351 = 'Distribution of Total Transactions'
_var350.set_title(_var351)
_var352 = 'class'
_var353 = df_5[_var352]
_var354 = 1
_var355 = (_var353 == _var354)
_var356 = df_5[_var355]
_var357 = 'day'
_var358 = _var356[_var357]
_var359 = 1
_var360 = axs_2[_var359]
sns.countplot(_var358, ax=_var360)
_var361 = 1
_var362 = axs_2[_var361]
_var363 = 'Distribution of Fraud Transactions'
_var362.set_title(_var363)
_var364 = 'class'
_var365 = df_5[_var364]
_var366 = 0
_var367 = (_var365 == _var366)
_var368 = df_5[_var367]
_var369 = 'day'
_var370 = _var368[_var369]
_var371 = 2
_var372 = axs_2[_var371]
sns.countplot(_var370, ax=_var372)
_var373 = 2
_var374 = axs_2[_var373]
_var375 = 'Distribution of Genuine Transactions'
_var374.set_title(_var375)
plt.show()
_var378 = 1
_var379 = 2
_var380 = 15
_var381 = 8
_var382 = (_var380, _var381)
(_var376, _var377) = plt.subplots(nrows=_var378, ncols=_var379, figsize=_var382)
fig_4 = _var376
axs_3 = _var377
_var383 = 'class'
_var384 = df_5[_var383]
_var385 = 0
_var386 = (_var384 == _var385)
_var387 = df_5[_var386]
_var388 = 'time'
_var389 = _var387[_var388]
_var390 = _var389.values
_var391 = 'green'
_var392 = 0
_var393 = axs_3[_var392]
sns.distplot(_var390, color=_var391, ax=_var393)
_var394 = 0
_var395 = axs_3[_var394]
_var396 = 'Genuine Transactions'
_var395.set_title(_var396)
_var397 = 'class'
_var398 = df_5[_var397]
_var399 = 1
_var400 = (_var398 == _var399)
_var401 = df_5[_var400]
_var402 = 'time'
_var403 = _var401[_var402]
_var404 = _var403.values
_var405 = 'red'
_var406 = 1
_var407 = axs_3[_var406]
sns.distplot(_var404, color=_var405, ax=_var407)
_var408 = 1
_var409 = axs_3[_var408]
_var410 = 'Fraud Transactions'
_var409.set_title(_var410)
_var411 = 'Comparison between Transaction Frequencies vs Time for Fraud and Genuine Transactions'
fig_4.suptitle(_var411)
plt.show()
_var412 = 12
_var413 = 10
_var414 = (_var412, _var413)
plt.figure(figsize=_var414)
_var415 = 'class'
_var416 = df_5[_var415]
_var417 = 0
_var418 = (_var416 == _var417)
_var419 = df_5[_var418]
_var420 = 'hour'
_var421 = _var419[_var420]
_var422 = 'green'
sns.distplot(_var421, color=_var422)
_var423 = 'class'
_var424 = df_5[_var423]
_var425 = 1
_var426 = (_var424 == _var425)
_var427 = df_5[_var426]
_var428 = 'hour'
_var429 = _var427[_var428]
_var430 = 'red'
sns.distplot(_var429, color=_var430)
_var431 = 'Fraud vs Genuine Transactions by Hours'
_var432 = 15
plt.title(_var431, fontsize=_var432)
_var433 = [0, 25]
plt.xlim(_var433)
plt.show()
_var434 = 8
_var435 = 6
_var436 = (_var434, _var435)
plt.figure(figsize=_var436)
_var437 = ['time', 'hour', 'day', 'amount', 'class']
_var438 = df_5[_var437]
_var439 = 'hour'
_var440 = _var438.groupby(_var439)
_var441 = _var440.count()
_var442 = 'class'
_var443 = _var441[_var442]
_var443.plot()
plt.show()
_var444 = 25
_var445 = 25
_var446 = (_var444, _var445)
df_5.hist(figsize=_var446)
plt.show()
_var447 = True
_var448 = True
df_5.reset_index(inplace=_var447, drop=_var448)
_var449 = 'amount_log'
_var450 = df_5.amount
_var451 = 0.01
_var452 = (_var450 + _var451)
_var453 = np.log(_var452)
df_6 = set_index_wrapper(df_5, _var449, _var453)
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
_var454 = 'amount_scaled'
_var455 = 'amount'
_var456 = df_6[_var455]
_var457 = _var456.values
_var458 = (- 1)
_var459 = 1
_var460 = _var457.reshape(_var458, _var459)
_var461 = ss.fit_transform(_var460)
df_7 = set_index_wrapper(df_6, _var454, _var461)
from sklearn.preprocessing import MinMaxScaler
mm = MinMaxScaler()
_var462 = 'amount_minmax'
_var463 = 'amount'
_var464 = df_7[_var463]
_var465 = _var464.values
_var466 = (- 1)
_var467 = 1
_var468 = _var465.reshape(_var466, _var467)
_var469 = mm.fit_transform(_var468)
df_8 = set_index_wrapper(df_7, _var462, _var469)
_var472 = 1
_var473 = 4
_var474 = 16
_var475 = 4
_var476 = (_var474, _var475)
(_var470, _var471) = plt.subplots(nrows=_var472, ncols=_var473, figsize=_var476)
fig_5 = _var470
axs_4 = _var471
_var477 = 'class'
_var478 = 'amount'
_var479 = 0
_var480 = axs_4[_var479]
sns.boxplot(x=_var477, y=_var478, data=df_8, ax=_var480)
_var481 = 0
_var482 = axs_4[_var481]
_var483 = 'Class vs Amount'
_var482.set_title(_var483)
_var484 = 'class'
_var485 = 'amount_log'
_var486 = 1
_var487 = axs_4[_var486]
sns.boxplot(x=_var484, y=_var485, data=df_8, ax=_var487)
_var488 = 1
_var489 = axs_4[_var488]
_var490 = 'Class vs Log Amount'
_var489.set_title(_var490)
_var491 = 'class'
_var492 = 'amount_scaled'
_var493 = 2
_var494 = axs_4[_var493]
sns.boxplot(x=_var491, y=_var492, data=df_8, ax=_var494)
_var495 = 2
_var496 = axs_4[_var495]
_var497 = 'Class vs Scaled Amount'
_var496.set_title(_var497)
_var498 = 'class'
_var499 = 'amount_minmax'
_var500 = 3
_var501 = axs_4[_var500]
sns.boxplot(x=_var498, y=_var499, data=df_8, ax=_var501)
_var502 = 3
_var503 = axs_4[_var502]
_var504 = 'Class vs Min Max Amount'
_var503.set_title(_var504)
plt.show()
_var505 = ['time', 'hour', 'day', 'amount', 'amount_log', 'amount_scaled', 'amount_minmax', 'class']
df_8[_var505]
CreditCardFraudDataCleaned = df_8
_var506 = 'CreditCardFraudDataCleaned.pkl'
_var507 = 'wb'
fileWriteStream = open(_var506, _var507)
with open(_var506, _var507) as fileWriteStream:
    pickle.dump(CreditCardFraudDataCleaned, fileWriteStream)
    fileWriteStream.close()
_var508 = 'pickle file is saved at Location:'
_var509 = os.getcwd()
_var510 = (_var508, _var509)
print(_var510)
_var511 = 'CreditCardFraudDataCleaned.pkl'
_var512 = 'rb'
fileReadStream = open(_var511, _var512)
with open(_var511, _var512) as fileReadStream:
    CreditCardFraudDataFromPickle = pickle.load(fileReadStream)
    fileReadStream.close()
df_9 = CreditCardFraudDataFromPickle
df_9.head()
df_9.shape
df_9.head()
df_9.columns
_var513 = ['time', 'class', 'hour', 'day', 'amount', 'amount_minmax', 'amount_scaled']
_var514 = 1
X = df_9.drop(_var513, axis=_var514)
_var515 = 'class'
y = df_9[_var515]
X
from sklearn.model_selection import train_test_split
_var520 = 0.3
_var521 = True
_var522 = 101
(_var516, _var517, _var518, _var519) = train_test_split(X, y, test_size=_var520, shuffle=_var521, random_state=_var522)
X_train = _var516
X_test = _var517
y_train = _var518
y_test = _var519
_var523 = 'X_train - '
_var524 = X_train.shape
_var525 = (_var523, _var524)
print(_var525)
_var526 = 'y_train - '
_var527 = y_train.shape
_var528 = (_var526, _var527)
print(_var528)
_var529 = 'X_test - '
_var530 = X_test.shape
_var531 = (_var529, _var530)
print(_var531)
_var532 = 'y_test - '
_var533 = y_test.shape
_var534 = (_var532, _var533)
print(_var534)
from sklearn.linear_model import LogisticRegression
_var539 = 0.3
_var540 = True
_var541 = 0
(_var535, _var536, _var537, _var538) = train_test_split(X, y, test_size=_var539, shuffle=_var540, random_state=_var541)
X_train_0 = _var535
X_test_0 = _var536
y_train_0 = _var537
y_test_0 = _var538
logreg = LogisticRegression()
logreg_0 = logreg.fit(X_train_0, y_train_0)
y_pred = logreg_0.predict(X_test_0)
from sklearn import metrics
_var542 = metrics.classification_report(y_test_0, y_pred)
print(_var542)
_var543 = 'Accuracy :{0:0.5f}'
_var544 = metrics.accuracy_score(y_pred, y_test_0)
_var545 = _var543.format(_var544)
print(_var545)
_var546 = 'AUC : {0:0.5f}'
_var547 = metrics.roc_auc_score(y_test_0, y_pred)
_var548 = _var546.format(_var547)
print(_var548)
_var549 = 'Precision : {0:0.5f}'
_var550 = metrics.precision_score(y_test_0, y_pred)
_var551 = _var549.format(_var550)
print(_var551)
_var552 = 'Recall : {0:0.5f}'
_var553 = metrics.recall_score(y_test_0, y_pred)
_var554 = _var552.format(_var553)
print(_var554)
_var555 = 'F1 : {0:0.5f}'
_var556 = metrics.f1_score(y_test_0, y_pred)
_var557 = _var555.format(_var556)
print(_var557)
_var558 = '\n'
print(_var558)
_var559 = pd.Series(y_pred)
_var559.value_counts()
_var560 = pd.Series(y_test_0)
_var560.value_counts()
_var561 = 103
_var562 = 147
(_var561 / _var562)
cnf_matrix = metrics.confusion_matrix(y_test_0, y_pred)
cnf_matrix
_var563 = pd.DataFrame(cnf_matrix)
_var564 = True
_var565 = {'size': 25}
_var566 = 'winter'
_var567 = 'g'
p = sns.heatmap(_var563, annot=_var564, annot_kws=_var565, cmap=_var566, fmt=_var567)
_var568 = 'Confusion matrix'
_var569 = 1.1
_var570 = 22
plt.title(_var568, y=_var569, fontsize=_var570)
_var571 = 'Actual'
_var572 = 18
plt.ylabel(_var571, fontsize=_var572)
_var573 = 'Predicted'
_var574 = 18
plt.xlabel(_var573, fontsize=_var574)
plt.show()
_var575 = 92
_var576 = 147
(_var575 / _var576)
metrics.roc_auc_score(y_test_0, y_pred)
y_pred_proba = logreg_0.predict_proba(X_test_0)
y_pred_proba
_var577 = 8
_var578 = 6
_var579 = (_var577, _var578)
plt.figure(figsize=_var579)
(_var580, _var581, _var582) = metrics.roc_curve(y_test_0, y_pred)
fpr = _var580
tpr = _var581
thresholds = _var582
auc = metrics.roc_auc_score(y_test_0, y_pred)
_var583 = 'AUC - '
_var584 = '\n'
_var585 = (_var583, auc, _var584)
print(_var585)
_var586 = 2
_var587 = 'data 1, auc='
_var588 = str(auc)
_var589 = (_var587 + _var588)
plt.plot(fpr, tpr, linewidth=_var586, label=_var589)
_var590 = 4
plt.legend(loc=_var590)
_var591 = [0, 1]
_var592 = [0, 1]
_var593 = 'k--'
plt.plot(_var591, _var592, _var593)
_var594 = plt.rcParams
_var595 = 'font.size'
_var596 = 12
_var594_0 = set_index_wrapper(_var594, _var595, _var596)
_var597 = 'ROC curve for Predicting a credit card fraud detection'
plt.title(_var597)
_var598 = 'False Positive Rate (1 - Specificity)'
plt.xlabel(_var598)
_var599 = 'True Positive Rate (Sensitivity)'
plt.ylabel(_var599)
plt.show()
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, classification_report
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
from sklearn.datasets import make_classification
_var600 = 'Original dataset shape %s'
_var601 = Counter(y)
_var602 = (_var600 % _var601)
print(_var602)
_var603 = 42
rus = RandomUnderSampler(random_state=_var603)
(_var604, _var605) = rus.fit_resample(X, y)
X_res = _var604
y_res = _var605
_var606 = 'Resampled dataset shape %s'
_var607 = Counter(y_res)
_var608 = (_var606 % _var607)
print(_var608)
_var613 = 0.3
_var614 = True
_var615 = 0
(_var609, _var610, _var611, _var612) = train_test_split(X_res, y_res, test_size=_var613, shuffle=_var614, random_state=_var615)
X_train_1 = _var609
X_test_1 = _var610
y_train_1 = _var611
y_test_1 = _var612
logreg_1 = LogisticRegression()
logreg_2 = logreg_1.fit(X_train_1, y_train_1)
y_pred_0 = logreg_2.predict(X_test_1)
_var616 = 'Accuracy :{0:0.5f}'
_var617 = metrics.accuracy_score(y_pred_0, y_test_1)
_var618 = _var616.format(_var617)
print(_var618)
_var619 = 'AUC : {0:0.5f}'
_var620 = metrics.roc_auc_score(y_test_1, y_pred_0)
_var621 = _var619.format(_var620)
print(_var621)
_var622 = 'Precision : {0:0.5f}'
_var623 = metrics.precision_score(y_test_1, y_pred_0)
_var624 = _var622.format(_var623)
print(_var624)
_var625 = 'Recall : {0:0.5f}'
_var626 = metrics.recall_score(y_test_1, y_pred_0)
_var627 = _var625.format(_var626)
print(_var627)
_var628 = 'F1 : {0:0.5f}'
_var629 = metrics.f1_score(y_test_1, y_pred_0)
_var630 = _var628.format(_var629)
print(_var630)
_var631 = 8
_var632 = 6
_var633 = (_var631, _var632)
plt.figure(figsize=_var633)
(_var634, _var635, _var636) = metrics.roc_curve(y_test_1, y_pred_0)
fpr_0 = _var634
tpr_0 = _var635
thresholds_0 = _var636
auc_0 = metrics.roc_auc_score(y_test_1, y_pred_0)
_var637 = 'AUC - '
_var638 = '\n'
_var639 = (_var637, auc_0, _var638)
print(_var639)
_var640 = 2
_var641 = 'data 1, auc='
_var642 = str(auc_0)
_var643 = (_var641 + _var642)
plt.plot(fpr_0, tpr_0, linewidth=_var640, label=_var643)
_var644 = 4
plt.legend(loc=_var644)
_var645 = [0, 1]
_var646 = [0, 1]
_var647 = 'k--'
plt.plot(_var645, _var646, _var647)
_var648 = plt.rcParams
_var649 = 'font.size'
_var650 = 12
_var648_0 = set_index_wrapper(_var648, _var649, _var650)
_var651 = 'ROC curve for Predicting a credit card fraud detection'
plt.title(_var651)
_var652 = 'False Positive Rate (1 - Specificity)'
plt.xlabel(_var652)
_var653 = 'True Positive Rate (Sensitivity)'
plt.ylabel(_var653)
plt.show()
cnf_matrix_0 = metrics.confusion_matrix(y_test_1, y_pred_0)
_var654 = pd.DataFrame(cnf_matrix_0)
_var655 = True
_var656 = {'size': 25}
_var657 = 'winter'
_var658 = 'g'
sns.heatmap(_var654, annot=_var655, annot_kws=_var656, cmap=_var657, fmt=_var658)
_var659 = 'Confusion matrix'
_var660 = 1.1
_var661 = 22
plt.title(_var659, y=_var660, fontsize=_var661)
_var662 = 'Predicted'
_var663 = 18
plt.xlabel(_var662, fontsize=_var663)
_var664 = 'Actual'
_var665 = 18
plt.ylabel(_var664, fontsize=_var665)
plt.show()
from imblearn.over_sampling import RandomOverSampler
_var666 = 'Original dataset shape %s'
_var667 = Counter(y)
_var668 = (_var666 % _var667)
print(_var668)
random_state = 42
ros = RandomOverSampler(random_state=random_state)
(_var669, _var670) = ros.fit_resample(X, y)
X_res_0 = _var669
y_res_0 = _var670
_var671 = 'Resampled dataset shape %s'
_var672 = Counter(y_res_0)
_var673 = (_var671 % _var672)
print(_var673)
_var678 = 0.3
_var679 = True
_var680 = 0
(_var674, _var675, _var676, _var677) = train_test_split(X_res_0, y_res_0, test_size=_var678, shuffle=_var679, random_state=_var680)
X_train_2 = _var674
X_test_2 = _var675
y_train_2 = _var676
y_test_2 = _var677
logreg_3 = LogisticRegression()
logreg_4 = logreg_3.fit(X_train_2, y_train_2)
y_pred_1 = logreg_4.predict(X_test_2)
_var681 = 'Accuracy :{0:0.5f}'
_var682 = metrics.accuracy_score(y_test_2, y_pred_1)
_var683 = _var681.format(_var682)
print(_var683)
_var684 = 'AUC : {0:0.5f}'
_var685 = metrics.roc_auc_score(y_test_2, y_pred_1)
_var686 = _var684.format(_var685)
print(_var686)
_var687 = 'Precision : {0:0.5f}'
_var688 = metrics.precision_score(y_test_2, y_pred_1)
_var689 = _var687.format(_var688)
print(_var689)
_var690 = 'Recall : {0:0.5f}'
_var691 = metrics.recall_score(y_test_2, y_pred_1)
_var692 = _var690.format(_var691)
print(_var692)
_var693 = 'F1 : {0:0.5f}'
_var694 = metrics.f1_score(y_test_2, y_pred_1)
_var695 = _var693.format(_var694)
print(_var695)
_var696 = 8
_var697 = 6
_var698 = (_var696, _var697)
plt.figure(figsize=_var698)
(_var699, _var700, _var701) = metrics.roc_curve(y_test_2, y_pred_1)
fpr_1 = _var699
tpr_1 = _var700
thresholds_1 = _var701
auc_1 = metrics.roc_auc_score(y_test_2, y_pred_1)
_var702 = 'AUC - '
_var703 = '\n'
_var704 = (_var702, auc_1, _var703)
print(_var704)
_var705 = 2
_var706 = 'data 1, auc='
_var707 = str(auc_1)
_var708 = (_var706 + _var707)
plt.plot(fpr_1, tpr_1, linewidth=_var705, label=_var708)
_var709 = 4
plt.legend(loc=_var709)
_var710 = [0, 1]
_var711 = [0, 1]
_var712 = 'k--'
plt.plot(_var710, _var711, _var712)
_var713 = plt.rcParams
_var714 = 'font.size'
_var715 = 12
_var713_0 = set_index_wrapper(_var713, _var714, _var715)
_var716 = 'ROC curve for Predicting a breast cancer classifier'
plt.title(_var716)
_var717 = 'False Positive Rate (1 - Specificity)'
plt.xlabel(_var717)
_var718 = 'True Positive Rate (Sensitivity)'
plt.ylabel(_var718)
plt.show()
cnf_matrix_1 = metrics.confusion_matrix(y_test_2, y_pred_1)
_var719 = pd.DataFrame(cnf_matrix_1)
_var720 = True
_var721 = {'size': 25}
_var722 = 'winter'
_var723 = 'g'
sns.heatmap(_var719, annot=_var720, annot_kws=_var721, cmap=_var722, fmt=_var723)
_var724 = 'Confusion matrix'
_var725 = 1.1
_var726 = 22
plt.title(_var724, y=_var725, fontsize=_var726)
_var727 = 'Predicted'
_var728 = 18
plt.xlabel(_var727, fontsize=_var728)
_var729 = 'Actual'
_var730 = 18
plt.ylabel(_var729, fontsize=_var730)
plt.show()
from imblearn.over_sampling import SMOTE, ADASYN
_var731 = 'Original dataset shape %s'
_var732 = Counter(y)
_var733 = (_var731 % _var732)
print(_var733)
_var734 = 42
smote = SMOTE(random_state=_var734)
(_var735, _var736) = smote.fit_resample(X, y)
X_res_1 = _var735
y_res_1 = _var736
_var737 = 'Resampled dataset shape %s'
_var738 = Counter(y_res_1)
_var739 = (_var737 % _var738)
print(_var739)
_var744 = 0.3
_var745 = True
_var746 = 0
(_var740, _var741, _var742, _var743) = train_test_split(X_res_1, y_res_1, test_size=_var744, shuffle=_var745, random_state=_var746)
X_train_3 = _var740
X_test_3 = _var741
y_train_3 = _var742
y_test_3 = _var743
_var747 = 1000
logreg_5 = LogisticRegression(max_iter=_var747)
logreg_6 = logreg_5.fit(X_train_3, y_train_3)
y_pred_2 = logreg_6.predict(X_test_3)
_var748 = 'Accuracy :{0:0.5f}'
_var749 = metrics.accuracy_score(y_test_3, y_pred_2)
_var750 = _var748.format(_var749)
print(_var750)
_var751 = 'AUC : {0:0.5f}'
_var752 = metrics.roc_auc_score(y_test_3, y_pred_2)
_var753 = _var751.format(_var752)
print(_var753)
_var754 = 'Precision : {0:0.5f}'
_var755 = metrics.precision_score(y_test_3, y_pred_2)
_var756 = _var754.format(_var755)
print(_var756)
_var757 = 'Recall : {0:0.5f}'
_var758 = metrics.recall_score(y_test_3, y_pred_2)
_var759 = _var757.format(_var758)
print(_var759)
_var760 = 'F1 : {0:0.5f}'
_var761 = metrics.f1_score(y_test_3, y_pred_2)
_var762 = _var760.format(_var761)
print(_var762)
_var763 = 8
_var764 = 6
_var765 = (_var763, _var764)
plt.figure(figsize=_var765)
(_var766, _var767, _var768) = metrics.roc_curve(y_test_3, y_pred_2)
fpr_2 = _var766
tpr_2 = _var767
thresholds_2 = _var768
auc_2 = metrics.roc_auc_score(y_test_3, y_pred_2)
_var769 = 'AUC - '
_var770 = '\n'
_var771 = (_var769, auc_2, _var770)
print(_var771)
_var772 = 2
_var773 = 'data 1, auc='
_var774 = str(auc_2)
_var775 = (_var773 + _var774)
plt.plot(fpr_2, tpr_2, linewidth=_var772, label=_var775)
_var776 = 4
plt.legend(loc=_var776)
_var777 = [0, 1]
_var778 = [0, 1]
_var779 = 'k--'
plt.plot(_var777, _var778, _var779)
_var780 = plt.rcParams
_var781 = 'font.size'
_var782 = 12
_var780_0 = set_index_wrapper(_var780, _var781, _var782)
_var783 = 'ROC curve for Predicting a breast cancer classifier'
plt.title(_var783)
_var784 = 'False Positive Rate (1 - Specificity)'
plt.xlabel(_var784)
_var785 = 'True Positive Rate (Sensitivity)'
plt.ylabel(_var785)
plt.show()
cnf_matrix_2 = metrics.confusion_matrix(y_test_3, y_pred_2)
_var786 = pd.DataFrame(cnf_matrix_2)
_var787 = True
_var788 = {'size': 25}
_var789 = 'winter'
_var790 = 'g'
sns.heatmap(_var786, annot=_var787, annot_kws=_var788, cmap=_var789, fmt=_var790)
_var791 = 'Confusion matrix'
_var792 = 1.1
_var793 = 22
plt.title(_var791, y=_var792, fontsize=_var793)
_var794 = 'Predicted'
_var795 = 18
plt.xlabel(_var794, fontsize=_var795)
_var796 = 'Actual'
_var797 = 18
plt.ylabel(_var796, fontsize=_var797)
plt.show()
_var798 = 'Original dataset shape %s'
_var799 = Counter(y)
_var800 = (_var798 % _var799)
print(_var800)
_var801 = 42
adasyn = ADASYN(random_state=_var801)
(_var802, _var803) = adasyn.fit_resample(X, y)
X_res_2 = _var802
y_res_2 = _var803
_var804 = 'Resampled dataset shape %s'
_var805 = Counter(y_res_2)
_var806 = (_var804 % _var805)
print(_var806)
_var811 = 0.3
_var812 = True
_var813 = 0
(_var807, _var808, _var809, _var810) = train_test_split(X_res_2, y_res_2, test_size=_var811, shuffle=_var812, random_state=_var813)
X_train_4 = _var807
X_test_4 = _var808
y_train_4 = _var809
y_test_4 = _var810
logreg_7 = LogisticRegression()
logreg_8 = logreg_7.fit(X_train_4, y_train_4)
y_pred_3 = logreg_8.predict(X_test_4)
_var814 = 'Accuracy :{0:0.5f}'
_var815 = metrics.accuracy_score(y_pred_3, y_test_4)
_var816 = _var814.format(_var815)
print(_var816)
_var817 = 'AUC : {0:0.5f}'
_var818 = metrics.roc_auc_score(y_test_4, y_pred_3)
_var819 = _var817.format(_var818)
print(_var819)
_var820 = 'Precision : {0:0.5f}'
_var821 = metrics.precision_score(y_test_4, y_pred_3)
_var822 = _var820.format(_var821)
print(_var822)
_var823 = 'Recall : {0:0.5f}'
_var824 = metrics.recall_score(y_test_4, y_pred_3)
_var825 = _var823.format(_var824)
print(_var825)
_var826 = 'F1 : {0:0.5f}'
_var827 = metrics.f1_score(y_test_4, y_pred_3)
_var828 = _var826.format(_var827)
print(_var828)
_var829 = 8
_var830 = 6
_var831 = (_var829, _var830)
plt.figure(figsize=_var831)
(_var832, _var833, _var834) = metrics.roc_curve(y_test_4, y_pred_3)
fpr_3 = _var832
tpr_3 = _var833
thresholds_3 = _var834
auc_3 = metrics.roc_auc_score(y_test_4, y_pred_3)
_var835 = 'AUC - '
_var836 = '\n'
_var837 = (_var835, auc_3, _var836)
print(_var837)
_var838 = 2
_var839 = 'data 1, auc='
_var840 = str(auc_3)
_var841 = (_var839 + _var840)
plt.plot(fpr_3, tpr_3, linewidth=_var838, label=_var841)
_var842 = 4
plt.legend(loc=_var842)
_var843 = [0, 1]
_var844 = [0, 1]
_var845 = 'k--'
plt.plot(_var843, _var844, _var845)
_var846 = plt.rcParams
_var847 = 'font.size'
_var848 = 12
_var846_0 = set_index_wrapper(_var846, _var847, _var848)
_var849 = 'ROC curve for Predicting a breast cancer classifier'
plt.title(_var849)
_var850 = 'False Positive Rate (1 - Specificity)'
plt.xlabel(_var850)
_var851 = 'True Positive Rate (Sensitivity)'
plt.ylabel(_var851)
plt.show()
cnf_matrix_3 = metrics.confusion_matrix(y_test_4, y_pred_3)
_var852 = pd.DataFrame(cnf_matrix_3)
_var853 = True
_var854 = {'size': 25}
_var855 = 'winter'
_var856 = 'g'
sns.heatmap(_var852, annot=_var853, annot_kws=_var854, cmap=_var855, fmt=_var856)
_var857 = 'Confusion matrix'
_var858 = 1.1
_var859 = 22
plt.title(_var857, y=_var858, fontsize=_var859)
_var860 = 'Predicted'
_var861 = 18
plt.xlabel(_var860, fontsize=_var861)
_var862 = 'Actual'
_var863 = 18
plt.ylabel(_var862, fontsize=_var863)
plt.show()
from sklearn.decomposition import PCA
_var864 = 2
_var865 = 42
_var866 = PCA(n_components=_var864, random_state=_var865)
X_reduced_pca_im = _var866.fit_transform(X)
_var867 = 12
_var868 = 8
_var869 = (_var867, _var868)
plt.figure(figsize=_var869)
_var870 = 0
_var871 = X_reduced_pca_im[:, _var870]
_var872 = 1
_var873 = X_reduced_pca_im[:, _var872]
_var874 = 0
_var875 = (y == _var874)
_var876 = 'No Fraud'
_var877 = 'coolwarm'
_var878 = 1
plt.scatter(_var871, _var873, c=_var875, label=_var876, cmap=_var877, linewidths=_var878)
_var879 = 0
_var880 = X_reduced_pca_im[:, _var879]
_var881 = 1
_var882 = X_reduced_pca_im[:, _var881]
_var883 = 1
_var884 = (y == _var883)
_var885 = 'Fraud'
_var886 = 'coolwarm'
_var887 = 1
plt.scatter(_var880, _var882, c=_var884, label=_var885, cmap=_var886, linewidths=_var887)
_var888 = 'Scatter Plot of Imbalanced Dataset'
plt.title(_var888)
plt.legend()
plt.show()
_var889 = 2
_var890 = 42
_var891 = PCA(n_components=_var889, random_state=_var890)
X_reduced_pca = _var891.fit_transform(X_res_2)
_var892 = 12
_var893 = 8
_var894 = (_var892, _var893)
plt.figure(figsize=_var894)
_var895 = 0
_var896 = X_reduced_pca[:, _var895]
_var897 = 1
_var898 = X_reduced_pca[:, _var897]
_var899 = 0
_var900 = (y_res_2 == _var899)
_var901 = 'coolwarm'
_var902 = 'No Fraud'
_var903 = 1
plt.scatter(_var896, _var898, c=_var900, cmap=_var901, label=_var902, linewidths=_var903)
_var904 = 0
_var905 = X_reduced_pca[:, _var904]
_var906 = 1
_var907 = X_reduced_pca[:, _var906]
_var908 = 1
_var909 = (y_res_2 == _var908)
_var910 = 'coolwarm'
_var911 = 'Fraud'
_var912 = 1
plt.scatter(_var905, _var907, c=_var909, cmap=_var910, label=_var911, linewidths=_var912)
_var913 = 'Scatter Plot of Imbalanced Dataset With Adaptive Synthetic Sampling \\(ADASYN\\)'
plt.title(_var913)
plt.legend()
plt.show()
_var914 = 'Original dataset shape %s'
_var915 = Counter(y)
_var916 = (_var914 % _var915)
print(_var916)
_var917 = 42
rus_0 = RandomUnderSampler(random_state=_var917)
(_var918, _var919) = rus_0.fit_resample(X, y)
X_under = _var918
y_under = _var919
_var920 = 'Resampled dataset shape %s'
_var921 = Counter(y_under)
_var922 = (_var920 % _var921)
print(_var922)
_var927 = True
_var928 = 0.3
_var929 = 0
(_var923, _var924, _var925, _var926) = train_test_split(X_under, y_under, shuffle=_var927, test_size=_var928, random_state=_var929)
X_train_under = _var923
X_test_under = _var924
y_train_under = _var925
y_test_under = _var926
_var930 = 'Original dataset shape %s'
_var931 = Counter(y)
_var932 = (_var930 % _var931)
print(_var932)
_var933 = 42
ros_0 = RandomOverSampler(random_state=_var933)
(_var934, _var935) = ros_0.fit_resample(X, y)
X_over = _var934
y_over = _var935
_var936 = 'Resampled dataset shape %s'
_var937 = Counter(y_over)
_var938 = (_var936 % _var937)
print(_var938)
_var943 = 0.3
_var944 = True
_var945 = 0
(_var939, _var940, _var941, _var942) = train_test_split(X_over, y_over, test_size=_var943, shuffle=_var944, random_state=_var945)
X_train_over = _var939
X_test_over = _var940
y_train_over = _var941
y_test_over = _var942
_var946 = 'Original dataset shape %s'
_var947 = Counter(y)
_var948 = (_var946 % _var947)
print(_var948)
_var949 = 42
smote_0 = SMOTE(random_state=_var949)
(_var950, _var951) = smote_0.fit_resample(X, y)
X_smote = _var950
y_smote = _var951
_var952 = 'Resampled dataset shape %s'
_var953 = Counter(y_smote)
_var954 = (_var952 % _var953)
print(_var954)
_var959 = 0.3
_var960 = True
_var961 = 0
(_var955, _var956, _var957, _var958) = train_test_split(X_smote, y_smote, test_size=_var959, shuffle=_var960, random_state=_var961)
X_train_smote = _var955
X_test_smote = _var956
y_train_smote = _var957
y_test_smote = _var958
_var962 = 'Original dataset shape %s'
_var963 = Counter(y)
_var964 = (_var962 % _var963)
print(_var964)
_var965 = 42
adasyn_0 = ADASYN(random_state=_var965)
(_var966, _var967) = adasyn_0.fit_resample(X, y)
X_adasyn = _var966
y_adasyn = _var967
_var968 = 'Resampled dataset shape %s'
_var969 = Counter(y_adasyn)
_var970 = (_var968 % _var969)
print(_var970)
_var975 = 0.3
_var976 = True
_var977 = 0
(_var971, _var972, _var973, _var974) = train_test_split(X_adasyn, y_adasyn, test_size=_var975, shuffle=_var976, random_state=_var977)
X_train_adasyn = _var971
X_test_adasyn = _var972
y_train_adasyn = _var973
y_test_adasyn = _var974
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
names_lst = []
aucs_train_lst = []
accuracy_train_lst = []
precision_train_lst = []
recall_train_lst = []
f1_train_lst = []
aucs_test_lst = []
accuracy_test_lst = []
precision_test_lst = []
recall_test_lst = []
f1_test_lst = []

def build_measure_model(models):
    _var978 = 12
    _var979 = 6
    _var980 = (_var978, _var979)
    plt.figure(figsize=_var980)
    for _var981 in models:
        _var986 = 0
        name = _var981[_var986]
        _var987 = 1
        model = _var981[_var987]
        _var988 = 2
        Xdata = _var981[_var988]
        _var989 = 3
        ydata = _var981[_var989]
        _var990 = global_wrapper(names_lst)
        _var990.append(name)
        _var995 = 0.3
        _var996 = True
        _var997 = 0
        (_var991, _var992, _var993, _var994) = train_test_split(Xdata, ydata, test_size=_var995, shuffle=_var996, random_state=_var997)
        X_train_5 = _var991
        X_test_5 = _var992
        y_train_5 = _var993
        y_test_5 = _var994
        model_0 = model.fit(X_train_5, y_train_5)
        y_train_pred = model_0.predict(X_train_5)
        y_test_pred = model_0.predict(X_test_5)
        Accuracy_train = metrics.accuracy_score(y_train_5, y_train_pred)
        _var998 = global_wrapper(accuracy_train_lst)
        _var998.append(Accuracy_train)
        Accuracy_test = metrics.accuracy_score(y_test_5, y_test_pred)
        _var999 = global_wrapper(accuracy_test_lst)
        _var999.append(Accuracy_test)
        Aucs_train = metrics.roc_auc_score(y_train_5, y_train_pred)
        _var1000 = global_wrapper(aucs_train_lst)
        _var1000.append(Aucs_train)
        Aucs_test = metrics.roc_auc_score(y_test_5, y_test_pred)
        _var1001 = global_wrapper(aucs_test_lst)
        _var1001.append(Aucs_test)
        PrecisionScore_train = metrics.precision_score(y_train_5, y_train_pred)
        _var1002 = global_wrapper(precision_train_lst)
        _var1002.append(PrecisionScore_train)
        PrecisionScore_test = metrics.precision_score(y_test_5, y_test_pred)
        _var1003 = global_wrapper(precision_test_lst)
        _var1003.append(PrecisionScore_test)
        RecallScore_train = metrics.recall_score(y_train_5, y_train_pred)
        _var1004 = global_wrapper(recall_train_lst)
        _var1004.append(RecallScore_train)
        RecallScore_test = metrics.recall_score(y_test_5, y_test_pred)
        _var1005 = global_wrapper(recall_test_lst)
        _var1005.append(RecallScore_test)
        F1Score_train = metrics.f1_score(y_train_5, y_train_pred)
        _var1006 = global_wrapper(f1_train_lst)
        _var1006.append(F1Score_train)
        F1Score_test = metrics.f1_score(y_test_5, y_test_pred)
        _var1007 = global_wrapper(f1_test_lst)
        _var1007.append(F1Score_test)
        cnf_matrix_4 = metrics.confusion_matrix(y_test_5, y_test_pred)
        _var1008 = 'Model Name :'
        _var1009 = (_var1008, name)
        print(_var1009)
        _var1010 = 'Train Accuracy :{0:0.5f}'
        _var1011 = _var1010.format(Accuracy_train)
        print(_var1011)
        _var1012 = 'Test Accuracy :{0:0.5f}'
        _var1013 = _var1012.format(Accuracy_test)
        print(_var1013)
        _var1014 = 'Train AUC : {0:0.5f}'
        _var1015 = _var1014.format(Aucs_train)
        print(_var1015)
        _var1016 = 'Test AUC : {0:0.5f}'
        _var1017 = _var1016.format(Aucs_test)
        print(_var1017)
        _var1018 = 'Train Precision : {0:0.5f}'
        _var1019 = _var1018.format(PrecisionScore_train)
        print(_var1019)
        _var1020 = 'Test Precision : {0:0.5f}'
        _var1021 = _var1020.format(PrecisionScore_test)
        print(_var1021)
        _var1022 = 'Train Recall : {0:0.5f}'
        _var1023 = _var1022.format(RecallScore_train)
        print(_var1023)
        _var1024 = 'Test Recall : {0:0.5f}'
        _var1025 = _var1024.format(RecallScore_test)
        print(_var1025)
        _var1026 = 'Train F1 : {0:0.5f}'
        _var1027 = _var1026.format(F1Score_train)
        print(_var1027)
        _var1028 = 'Test F1 : {0:0.5f}'
        _var1029 = _var1028.format(F1Score_test)
        print(_var1029)
        _var1030 = 'Confusion Matrix : \n'
        _var1031 = (_var1030, cnf_matrix_4)
        print(_var1031)
        _var1032 = '\n'
        print(_var1032)
        (_var1033, _var1034, _var1035) = metrics.roc_curve(y_test_5, y_test_pred)
        fpr_4 = _var1033
        tpr_4 = _var1034
        thresholds_4 = _var1035
        auc_4 = metrics.roc_auc_score(y_test_5, y_test_pred)
        _var1036 = 2
        _var1037 = ', auc='
        _var1038 = (name + _var1037)
        _var1039 = str(auc_4)
        _var1040 = (_var1038 + _var1039)
        plt.plot(fpr_4, tpr_4, linewidth=_var1036, label=_var1040)
    thresholds_5 = __phi__(thresholds_4, thresholds_3)
    cnf_matrix_5 = __phi__(cnf_matrix_4, cnf_matrix_3)
    y_train_6 = __phi__(y_train_5, y_train_4)
    y_test_6 = __phi__(y_test_5, y_test_4)
    fpr_5 = __phi__(fpr_4, fpr_3)
    auc_5 = __phi__(auc_4, auc_3)
    X_train_6 = __phi__(X_train_5, X_train_4)
    tpr_5 = __phi__(tpr_4, tpr_3)
    X_test_6 = __phi__(X_test_5, X_test_4)
    _var1041 = 4
    plt.legend(loc=_var1041)
    _var1042 = [0, 1]
    _var1043 = [0, 1]
    _var1044 = 'k--'
    plt.plot(_var1042, _var1043, _var1044)
    _var1045 = plt.rcParams
    _var1046 = 'font.size'
    _var1047 = 12
    _var1045_0 = set_index_wrapper(_var1045, _var1046, _var1047)
    _var1048 = 'ROC curve for Predicting a credit card fraud detection'
    plt.title(_var1048)
    _var1049 = 'False Positive Rate (1 - Specificity)'
    plt.xlabel(_var1049)
    _var1050 = 'True Positive Rate (Sensitivity)'
    plt.ylabel(_var1050)
    plt.show()
LRmodels = []
_var1051 = 'LR imbalance'
_var1052 = 'liblinear'
_var1053 = 'ovr'
_var1054 = LogisticRegression(solver=_var1052, multi_class=_var1053)
_var1055 = (_var1051, _var1054, X, y)
LRmodels.append(_var1055)
_var1056 = 'LR Undersampling'
_var1057 = 'liblinear'
_var1058 = 'ovr'
_var1059 = LogisticRegression(solver=_var1057, multi_class=_var1058)
_var1060 = (_var1056, _var1059, X_under, y_under)
LRmodels.append(_var1060)
_var1061 = 'LR Oversampling'
_var1062 = 'liblinear'
_var1063 = 'ovr'
_var1064 = LogisticRegression(solver=_var1062, multi_class=_var1063)
_var1065 = (_var1061, _var1064, X_over, y_over)
LRmodels.append(_var1065)
_var1066 = 'LR SMOTE'
_var1067 = 'liblinear'
_var1068 = 'ovr'
_var1069 = LogisticRegression(solver=_var1067, multi_class=_var1068)
_var1070 = (_var1066, _var1069, X_smote, y_smote)
LRmodels.append(_var1070)
_var1071 = 'LR ADASYN'
_var1072 = 'liblinear'
_var1073 = 'ovr'
_var1074 = LogisticRegression(solver=_var1072, multi_class=_var1073)
_var1075 = (_var1071, _var1074, X_adasyn, y_adasyn)
LRmodels.append(_var1075)
build_measure_model(LRmodels)
DTmodels = []
dt = DecisionTreeClassifier()
_var1076 = 'DT imbalance'
_var1077 = (_var1076, dt, X, y)
DTmodels.append(_var1077)
_var1078 = 'DT Undersampling'
_var1079 = (_var1078, dt, X_under, y_under)
DTmodels.append(_var1079)
_var1080 = 'DT Oversampling'
_var1081 = (_var1080, dt, X_over, y_over)
DTmodels.append(_var1081)
_var1082 = 'DT SMOTE'
_var1083 = (_var1082, dt, X_smote, y_smote)
DTmodels.append(_var1083)
_var1084 = 'DT ADASYN'
_var1085 = (_var1084, dt, X_adasyn, y_adasyn)
DTmodels.append(_var1085)
build_measure_model(DTmodels)
RFmodels = []
_var1086 = 'RF imbalance'
_var1087 = RandomForestClassifier()
_var1088 = (_var1086, _var1087, X, y)
RFmodels.append(_var1088)
_var1089 = 'RF Undersampling'
_var1090 = RandomForestClassifier()
_var1091 = (_var1089, _var1090, X_under, y_under)
RFmodels.append(_var1091)
_var1092 = 'RF Oversampling'
_var1093 = RandomForestClassifier()
_var1094 = (_var1092, _var1093, X_over, y_over)
RFmodels.append(_var1094)
_var1095 = 'RF SMOTE'
_var1096 = RandomForestClassifier()
_var1097 = (_var1095, _var1096, X_smote, y_smote)
RFmodels.append(_var1097)
_var1098 = 'RF ADASYN'
_var1099 = RandomForestClassifier()
_var1100 = (_var1098, _var1099, X_adasyn, y_adasyn)
RFmodels.append(_var1100)
build_measure_model(RFmodels)
NBmodels = []
_var1101 = 'NB imbalance'
_var1102 = GaussianNB()
_var1103 = (_var1101, _var1102, X, y)
NBmodels.append(_var1103)
_var1104 = 'NB Undersampling'
_var1105 = GaussianNB()
_var1106 = (_var1104, _var1105, X_under, y_under)
NBmodels.append(_var1106)
_var1107 = 'NB Oversampling'
_var1108 = GaussianNB()
_var1109 = (_var1107, _var1108, X_over, y_over)
NBmodels.append(_var1109)
_var1110 = 'NB SMOTE'
_var1111 = GaussianNB()
_var1112 = (_var1110, _var1111, X_smote, y_smote)
NBmodels.append(_var1112)
_var1113 = 'NB ADASYN'
_var1114 = GaussianNB()
_var1115 = (_var1113, _var1114, X_adasyn, y_adasyn)
NBmodels.append(_var1115)
build_measure_model(NBmodels)
_var1116 = global_wrapper(names_lst)
_var1117 = global_wrapper(accuracy_train_lst)
_var1118 = global_wrapper(accuracy_test_lst)
_var1119 = global_wrapper(aucs_train_lst)
_var1120 = global_wrapper(aucs_test_lst)
_var1121 = global_wrapper(precision_train_lst)
_var1122 = global_wrapper(precision_test_lst)
_var1123 = global_wrapper(recall_train_lst)
_var1124 = global_wrapper(recall_test_lst)
_var1125 = global_wrapper(f1_train_lst)
_var1126 = global_wrapper(f1_test_lst)
data = {'Model': _var1116, 'Accuracy_Train': _var1117, 'Accuracy_Test': _var1118, 'AUC_Train': _var1119, 'AUC_Test': _var1120, 'PrecisionScore_Train': _var1121, 'PrecisionScore_Test': _var1122, 'RecallScore_Train': _var1123, 'RecallScore_Test': _var1124, 'F1Score_Train': _var1125, 'F1Score_Test': _var1126}
_var1127 = 'Performance measures of various classifiers: \n'
print(_var1127)
performance_df = pd.DataFrame(data)
_var1128 = ['AUC_Test', 'RecallScore_Test', 'F1Score_Test']
_var1129 = False
performance_df.sort_values(_var1128, ascending=_var1129)
_var1130 = 'Gol_qOgRqfA'
_var1131 = 800
_var1132 = 400
YouTubeVideo(_var1130, width=_var1131, height=_var1132)
from sklearn.model_selection import GridSearchCV
_var1133 = ['saga']
_var1134 = ['l1', 'l2']
_var1135 = [0.01, 0.1, 1, 10, 100]
_var1136 = [100000]
_var1137 = {'solver': _var1133, 'penalty': _var1134, 'C': _var1135, 'max_iter': _var1136}
log_reg_params = (_var1137,)
_var1138 = LogisticRegression()
grid_log_reg = GridSearchCV(_var1138, log_reg_params)
grid_log_reg_0 = grid_log_reg.fit(X_train_under, y_train_under)
_var1139 = 'Logistic Regression best estimator : \n'
_var1140 = grid_log_reg_0.best_estimator_
_var1141 = (_var1139, _var1140)
print(_var1141)
y_pred_lr = grid_log_reg_0.predict(X_test_under)
_var1142 = '\nLogistic Regression f1 Score : {0:0.5f}'
_var1143 = metrics.f1_score(y_test_under, y_pred_lr)
_var1144 = _var1142.format(_var1143)
print(_var1144)
_var1145 = 2
_var1146 = 60
_var1147 = 1
_var1148 = range(_var1145, _var1146, _var1147)
_var1149 = list(_var1148)
_var1150 = ['auto', 'ball_tree', 'kd_tree', 'brute']
knears_params = {'n_neighbors': _var1149, 'algorithm': _var1150}
_var1151 = KNeighborsClassifier()
grid_knears = GridSearchCV(_var1151, knears_params)
grid_knears_0 = grid_knears.fit(X_train_under, y_train_under)
_var1152 = 'KNN best estimator : \n'
_var1153 = grid_knears_0.best_estimator_
_var1154 = (_var1152, _var1153)
print(_var1154)
y_pred_knn = grid_knears_0.predict(X_test_under)
_var1155 = '\nKNN f1 Score : {0:0.5f}'
_var1156 = metrics.f1_score(y_test_under, y_pred_knn)
_var1157 = _var1155.format(_var1156)
print(_var1157)
_var1158 = [0.5, 0.7, 0.9, 1]
_var1159 = ['rbf', 'poly', 'sigmoid', 'linear']
svc_params = {'C': _var1158, 'kernel': _var1159}
_var1160 = SVC()
grid_svc = GridSearchCV(_var1160, svc_params)
grid_svc_0 = grid_svc.fit(X_train_under, y_train_under)
_var1161 = 'SVC best estimator : \n'
_var1162 = grid_svc_0.best_estimator_
_var1163 = (_var1161, _var1162)
print(_var1163)
y_pred_svc = grid_svc_0.predict(X_test_under)
_var1164 = '\nSVC f1 Score : {0:0.5f}'
_var1165 = metrics.f1_score(y_test_under, y_pred_svc)
_var1166 = _var1164.format(_var1165)
print(_var1166)
_var1167 = ['gini', 'entropy']
_var1168 = 2
_var1169 = 4
_var1170 = 1
_var1171 = range(_var1168, _var1169, _var1170)
_var1172 = list(_var1171)
_var1173 = 5
_var1174 = 7
_var1175 = 1
_var1176 = range(_var1173, _var1174, _var1175)
_var1177 = list(_var1176)
tree_params = {'criterion': _var1167, 'max_depth': _var1172, 'min_samples_leaf': _var1177}
_var1178 = DecisionTreeClassifier()
_var1179 = 'accuracy'
_var1180 = 5
_var1181 = 1
_var1182 = (- 1)
grid_tree = GridSearchCV(estimator=_var1178, param_grid=tree_params, scoring=_var1179, cv=_var1180, verbose=_var1181, n_jobs=_var1182)
grid_tree_0 = grid_tree.fit(X_train_under, y_train_under)
_var1183 = 'Decision Tree best estimator : \n'
_var1184 = grid_tree_0.best_estimator_
_var1185 = (_var1183, _var1184)
print(_var1185)
y_pred_dt = grid_tree_0.predict(X_test_under)
_var1186 = '\nf1 Score : {0:0.5f}'
_var1187 = metrics.f1_score(y_test_under, y_pred_dt)
_var1188 = _var1186.format(_var1187)
print(_var1188)
