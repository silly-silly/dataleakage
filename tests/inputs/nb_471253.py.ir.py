

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
import pandas as pd
import numpy as np
import seaborn as sns
from collections import Counter
import matplotlib as mpl
import matplotlib.pyplot as plt
import pdb
_var3 = 'data/dataset_challenge_one (6).tsv'
_var4 = '\t'
df = pd.read_csv(_var3, sep=_var4)
df.head()
df.shape
df.info()
_var5 = df.iloc
_var6 = 10
_var7 = _var5[:, :_var6]
_var7.describe()
_var8 = 'class'
_var9 = df[_var8]
Counter(_var9)
_var10 = 'Pre-Clean: {}'
_var11 = df.shape
_var12 = _var10.format(_var11)
print(_var12)
_var13 = True
df.dropna(inplace=_var13)
_var14 = 'Post-Clean: {}'
_var15 = df.shape
_var16 = _var14.format(_var15)
print(_var16)
_var17 = 'class'
_var18 = 1
df_vars = df.drop(_var17, axis=_var18)
_var19 = df_vars.describe()
_var20 = _var19.loc
_var21 = ['min', 'max', 'mean', 'std']
summary_stats = _var20[_var21]
summary_stats_0 = summary_stats.transpose()
_var22 = 'range'
_var23 = 'max'
_var24 = summary_stats_0[_var23]
_var25 = 'min'
_var26 = summary_stats_0[_var25]
_var27 = (_var24 - _var26)
summary_stats_1 = set_index_wrapper(summary_stats_0, _var22, _var27)
summary_stats_1.head()
_var30 = 2
_var31 = 2
_var32 = 20
_var33 = 12
_var34 = (_var32, _var33)
(_var28, _var29) = plt.subplots(_var30, _var31, figsize=_var34)
f = _var28
_var39 = 0
_var40 = _var29[_var39]
_var41 = 0
ax1 = _var40[_var41]
_var42 = 1
ax2 = _var40[_var42]
_var45 = 1
_var46 = _var29[_var45]
_var47 = 0
ax3 = _var46[_var47]
_var48 = 1
ax4 = _var46[_var48]
_var49 = global_wrapper(ax1)
_var50 = 'Distribution of Variable Min.'
_var51 = 16
_var49.set_title(_var50, size=_var51)
_var52 = global_wrapper(ax2)
_var53 = 'Distribution of Variable Max.'
_var54 = 16
_var52.set_title(_var53, size=_var54)
_var55 = global_wrapper(ax3)
_var56 = 'Distribution of Variable Means'
_var57 = 16
_var55.set_title(_var56, size=_var57)
_var58 = global_wrapper(ax4)
_var59 = 'Distribution of Variable Variances'
_var60 = 16
_var58.set_title(_var59, size=_var60)
_var61 = 'min'
_var62 = summary_stats_1[_var61]
_var63 = False
_var64 = global_wrapper(ax1)
sns.distplot(_var62, axlabel=_var63, ax=_var64)
_var65 = 'max'
_var66 = summary_stats_1[_var65]
_var67 = False
_var68 = global_wrapper(ax2)
sns.distplot(_var66, axlabel=_var67, ax=_var68)
_var69 = 'class'
_var70 = df[_var69]
_var71 = 0
_var72 = (_var70 == _var71)
_var73 = df_vars[_var72]
_var74 = _var73.mean()
_var75 = 20
_var76 = False
_var77 = 'Class=0'
_var78 = global_wrapper(ax3)
sns.distplot(_var74, bins=_var75, axlabel=_var76, label=_var77, ax=_var78)
_var79 = 'class'
_var80 = df[_var79]
_var81 = 1
_var82 = (_var80 == _var81)
_var83 = df_vars[_var82]
_var84 = _var83.mean()
_var85 = 20
_var86 = False
_var87 = 'Class=1'
_var88 = global_wrapper(ax3)
sns.distplot(_var84, bins=_var85, axlabel=_var86, label=_var87, ax=_var88)
_var89 = global_wrapper(ax3)
_var89.legend()
_var90 = 'std'
_var91 = summary_stats_1[_var90]

def _func0(x):
    _var92 = 2
    _var93 = (x ** _var92)
    return _var93
_var94 = _var91.map(_func0)
_var95 = False
_var96 = global_wrapper(ax4)
sns.distplot(_var94, axlabel=_var95, ax=_var96)
_var99 = 1
_var100 = 2
_var101 = 20
_var102 = 6
_var103 = (_var101, _var102)
(_var97, _var98) = plt.subplots(_var99, _var100, figsize=_var103)
f_0 = _var97
_var106 = 0
ax5 = _var98[_var106]
_var107 = 1
ax6 = _var98[_var107]
_var108 = global_wrapper(ax5)
_var109 = 'Distribution of All Variables (Using Kernel Density Estimate, Alpha=0.1)'
_var110 = 16
_var108.set_title(_var109, size=_var110)
_var111 = global_wrapper(ax6)
_var112 = 'Distribution of All Variables (Using Kernel Density Estimate, Alpha=0.01)'
_var113 = 16
_var111.set_title(_var112, size=_var113)
_var114 = df_vars.columns
for column in _var114:
    _var115 = df_vars[column]
    _var116 = 'gray'
    _var117 = False
    _var118 = False
    _var119 = global_wrapper(ax5)
    _var120 = {'alpha': 0.1}
    sns.distplot(_var115, color=_var116, hist=_var117, axlabel=_var118, ax=_var119, kde_kws=_var120)
    _var121 = df_vars[column]
    _var122 = 'gray'
    _var123 = False
    _var124 = False
    _var125 = global_wrapper(ax6)
    _var126 = {'alpha': 0.01}
    sns.distplot(_var121, color=_var122, hist=_var123, axlabel=_var124, ax=_var125, kde_kws=_var126)
_var127 = df.ix
_var128 = df.columns
_var129 = 'class'
_var130 = (_var128 != _var129)
_var131 = _var127[:, _var130]
_var132 = 'class'
_var133 = df[_var132]
corr_versus_class = _var131.corrwith(_var133)
_var134 = 20
_var135 = 6
_var136 = (_var134, _var135)
plt.figure(figsize=_var136)
_var137 = 'Correlation between Variable vs. Class Label Distribution'
_var138 = 20
plt.title(_var137, size=_var138)
_var139 = 'Variable and Class Label Correlations'
_var140 = 16
plt.xlabel(_var139, size=_var140)
sns.distplot(corr_versus_class)
_var141 = 50
np.percentile(corr_versus_class, _var141)
_var142 = 'class'
labels = df[_var142]
_var143 = 'class'
_var144 = 1
df_vars_0 = df.drop(_var143, axis=_var144)
_var145 = df_vars_0.iloc
_var146 = corr_versus_class.argsort()
_var147 = (- 1)
_var148 = _var146[::_var147]
df_vars_1 = _var145[:, _var148]
_var149 = 20
_var150 = 10
_var151 = (_var149, _var150)
plt.figure(figsize=_var151)
_var152 = 'Correlation Matrix Heatmap between all Variables'
_var153 = 16
plt.title(_var152, size=_var153)
_var154 = df_vars_1.corr()
_var155 = False
_var156 = False
sns.heatmap(_var154, xticklabels=_var155, yticklabels=_var156)
from scipy.stats import probplot
corr_coeffs = []
_var157 = df_vars_1.columns
for column_0 in _var157:
    _var160 = df_vars_1[column_0]
    _var161 = 'norm'
    (_var158, _var159) = probplot(_var160, dist=_var161)
    _ = _var158
    _var165 = 0
    slope = _var159[_var165]
    _var166 = 1
    intercept = _var159[_var166]
    _var167 = 2
    r = _var159[_var167]
    corr_coeffs.append(r)
_var168 = df_vars_1.iloc
_var169 = np.argsort(corr_coeffs)
_var170 = 0
_var171 = _var169[_var170]
_var172 = _var168[:, _var171]
_var173 = 'norm'
res = probplot(_var172, dist=_var173, plot=plt)
_var174 = 'Quantile-Quantile Plot'
_var175 = 16
plt.title(_var174, size=_var175)
_var176 = 'Theoretical Quantile'
plt.xlabel(_var176)
_var177 = 'Sample Quantile'
plt.ylabel(_var177)
_var178 = 20
_var179 = 5
_var180 = (_var178, _var179)
plt.figure(figsize=_var180)
_var181 = 'Distributions of Lowest Correlated QQ-plots'
_var182 = 18
plt.title(_var181, size=_var182)
_var183 = np.argsort(corr_coeffs)
_var184 = 20
anomaly_indices = _var183[:_var184]
for i in anomaly_indices:
    _var185 = df_vars_1.iloc
    _var186 = _var185[:, i]
    _var187 = False
    sns.distplot(_var186, hist=_var187)
from sklearn.decomposition import PCA
_var188 = 2
pca = PCA(n_components=_var188)
_var189 = df_vars_1.values
reduced_data = pca.fit_transform(_var189)
pca.explained_variance_ratio_
_var190 = ['PC1', 'PC2']
df_clusters = pd.DataFrame(reduced_data, columns=_var190)
_var191 = 'class'
df_clusters_0 = set_index_wrapper(df_clusters, _var191, labels)
_var192 = 'PC1'
_var193 = 'PC2'
_var194 = 'class'
_var195 = 8
_var196 = False
sns.lmplot(x=_var192, y=_var193, hue=_var194, data=df_clusters_0, size=_var195, fit_reg=_var196)
_var197 = 'PCA (Components=2)'
_var198 = 16
plt.title(_var197, size=_var198)
_var199 = corr_versus_class.max()
_var200 = corr_versus_class.min()
(_var199, _var200)
sig_var_name = corr_versus_class.argmax()
_var201 = 'Variable with most significant statistic: {}'
_var201.format(sig_var_name)
_var202 = corr_versus_class.argmax()
_var203 = df[_var202]
_df = pd.DataFrame(_var203)
_var204 = 'class'
_df_0 = set_index_wrapper(_df, _var204, labels)
_var205 = ['variable', 'class']
_df_1 = set_field_wrapper(_df_0, 'columns', _var205)
_var206 = True
_df_1.dropna(inplace=_var206)
_df_1.head()
_var207 = 16
_var208 = 5
_var209 = (_var207, _var208)
_var210 = {'figure.figsize': _var209}
sns.set(rc=_var210)
_var211 = 'class'
_var212 = _df_1[_var211]
_var213 = 0
_var214 = (_var212 == _var213)
_var215 = _df_1[_var214]
_var216 = _var215.variable
_var217 = 'class=0'
g = sns.distplot(_var216, label=_var217)
_var218 = 'class'
_var219 = _df_1[_var218]
_var220 = 1
_var221 = (_var219 == _var220)
_var222 = _df_1[_var221]
_var223 = _var222.variable
_var224 = 'class=1'
sns.distplot(_var223, label=_var224, ax=g, axlabel=sig_var_name)
_var225 = g.legend()
_var226 = True
_var225.set_visible(_var226)
_var227 = 'Distribution Plots for Variable 1497 (by Class)'
_var228 = 16
plt.title(_var227, size=_var228)
_var229 = 6
_var230 = 6
_var231 = (_var229, _var230)
_var232 = {'figure.figsize': _var231}
sns.set(rc=_var232)
_var233 = 'class'
_var234 = 'variable'
sns.boxplot(x=_var233, y=_var234, data=_df_1)
_var235 = 'Boxplots for Variable 1497 (by Class)'
_var236 = 12
plt.title(_var235, size=_var236)
_var237 = 'class'
_var238 = 'variable'
sns.swarmplot(x=_var237, y=_var238, data=_df_1)
_var239 = 'Swarmplots for Variable 1497 (by Class)'
_var240 = 12
plt.title(_var239, size=_var240)
_var241 = df_clusters_0.PC1
_var242 = df_vars_1.corrwith(_var241)

def _func1(x_0):
    _var243 = 2
    _var244 = (x_0 ** _var243)
    return _var244
rsquared_with_pca = _var242.map(_func1)
_var245 = 'Highest R-Squared: {} - {} and PC1'
_var246 = rsquared_with_pca.max()
_var247 = np.argmax(rsquared_with_pca)
_var245.format(_var246, _var247)
sig_var_name_0 = rsquared_with_pca.argmax()
_var248 = 'Variable with most significant statistic: {}'
_var248.format(sig_var_name_0)
_df_2 = None
_var249 = df[sig_var_name_0]
_df_3 = pd.DataFrame(_var249)
_var250 = 'pc1'
_var251 = df_clusters_0.PC1
_df_4 = set_index_wrapper(_df_3, _var250, _var251)
_var252 = 'class'
_df_5 = set_index_wrapper(_df_4, _var252, labels)
_var253 = ['variable', 'pc1', 'class']
_df_6 = set_field_wrapper(_df_5, 'columns', _var253)
_var254 = True
_df_6.dropna(inplace=_var254)
_df_6.head()
_var255 = 'pc1'
_var256 = 'variable'
sns.lmplot(x=_var255, y=_var256, data=_df_6)
_var257 = 'Scatterplot - Variable 578 vs. PC1'
plt.title(_var257)
_var258 = 'pc1'
_var259 = 'variable'
_var260 = 'class'
_var261 = False
sns.lmplot(x=_var258, y=_var259, hue=_var260, data=_df_6, fit_reg=_var261)
_var262 = 'Scatterplot - Variable 578 vs. PC1 (by Class)'
plt.title(_var262)
_var263 = 10
_var264 = 4
_var265 = (_var263, _var264)
_var266 = {'figure.figsize': _var265}
sns.set(rc=_var266)
_var267 = 'pc1'
_var268 = 'variable'
sns.residplot(_var267, _var268, data=_df_6)
_var269 = 'Residual Plot for Variable 578 vs. PC1'
_var270 = 16
plt.title(_var269, size=_var270)
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from itertools import product
_var275 = df_vars_1.values
_var276 = labels.values
_var277 = 0.3
_var278 = 5
(_var271, _var272, _var273, _var274) = train_test_split(_var275, _var276, test_size=_var277, random_state=_var278)
X_train = _var271
X_test = _var272
y_train = _var273
y_test = _var274
_var279 = X_train.shape
_var280 = X_test.shape
_var281 = y_train.shape
_var282 = y_test.shape
(_var279, _var280, _var281, _var282)
_var283 = X_train.shape
_var284 = 0
_var285 = _var283[_var284]
_var286 = y_train.shape
_var287 = 0
_var288 = _var286[_var287]
_var289 = (_var285 == _var288)
assert _var289
_var290 = X_test.shape
_var291 = 0
_var292 = _var290[_var291]
_var293 = y_test.shape
_var294 = 0
_var295 = _var293[_var294]
_var296 = (_var292 == _var295)
assert _var296
verbose = False
_var297 = 4
_var298 = True
_var299 = 5
kfold = StratifiedKFold(y=y_train, n_folds=_var297, shuffle=_var298, random_state=_var299)
_var300 = ['hinge']
_var301 = [0.001, 0.01, 0.1, 1]
_var302 = 0
_var303 = 1.2
_var304 = 0.2
_var305 = np.arange(_var302, _var303, _var304)
_var306 = list(_var305)
_var307 = 2
_var308 = 30
_var309 = 2
_var310 = range(_var307, _var308, _var309)
_var311 = list(_var310)
params_range = {'loss': _var300, 'alpha': _var301, 'l1_ratio': _var306, 'n_components': _var311}
results = []
_var312 = 'loss'
_var313 = params_range[_var312]
_var314 = 'alpha'
_var315 = params_range[_var314]
_var316 = 'l1_ratio'
_var317 = params_range[_var316]
_var318 = 'n_components'
_var319 = params_range[_var318]
_var320 = product(_var313, _var315, _var317, _var319)
for _var321 in _var320:
    _var326 = 0
    loss = _var321[_var326]
    _var327 = 1
    alpha = _var321[_var327]
    _var328 = 2
    l1_ratio = _var321[_var328]
    _var329 = 3
    n_component = _var321[_var329]
    pca_0 = PCA(n_components=n_component)
    reduced_data_0 = pca_0.fit_transform(X_train)
    _var330 = 'elasticnet'
    _var331 = 1000
    clf = SGDClassifier(loss=loss, penalty=_var330, alpha=alpha, l1_ratio=l1_ratio, n_iter=_var331)
    _var332 = []
    _var333 = []
    scores_accuracy = _var332
    scores_f1 = _var333
    _var334 = enumerate(kfold)
    for _var335 in _var334:
        _var338 = 0
        k = _var335[_var338]
        _var341 = 1
        _var342 = _var335[_var341]
        _var343 = 0
        train_index = _var342[_var343]
        _var344 = 1
        cv_index = _var342[_var344]
        _var345 = reduced_data_0[train_index]
        _var346 = y_train[train_index]
        clf_0 = clf.fit(_var345, _var346)
        _var347 = reduced_data_0[cv_index]
        pred = clf_0.predict(_var347)
        _var348 = y_train[cv_index]
        _var349 = accuracy_score(_var348, pred)
        _var350 = 3
        score_accuracy = round(_var349, _var350)
        _var351 = y_train[cv_index]
        _var352 = f1_score(_var351, pred)
        _var353 = 3
        score_f1 = round(_var352, _var353)
        scores_accuracy.append(score_accuracy)
        scores_f1.append(score_f1)
        if verbose:
            _var354 = 'Fold: {}, Accuracy: {}, F1-Score: {}, Alpha: {}, L1-ratio: {}, Loss: {}, PCA Components: {}'
            _var355 = _var354.format(k, score_accuracy, score_f1, alpha, l1_ratio, loss, n_component)
            print(_var355)
    clf_1 = __phi__(clf_0, clf)
    _var356 = np.mean(scores_accuracy)
    _var357 = 3
    _var358 = round(_var356, _var357)
    _var359 = np.mean(scores_f1)
    _var360 = 3
    _var361 = round(_var359, _var360)
    _var362 = (_var358, _var361, alpha, l1_ratio, loss, n_component)
    results.append(_var362)
reduced_data_1 = __phi__(reduced_data_0, reduced_data)
pca_1 = __phi__(pca_0, pca)
_var363 = 'Best Accuracy: {}'

def _func2(x_1):
    _var364 = 0
    _var365 = x_1[_var364]
    return _var365
_var366 = max(results, key=_func2)
_var367 = _var363.format(_var366)
print(_var367)
_var368 = 'Best F1-Score: {}'

def _func3(x_2):
    _var369 = 1
    _var370 = x_2[_var369]
    return _var370
_var371 = max(results, key=_func3)
_var372 = _var368.format(_var371)
print(_var372)
_var373 = 18
pca_2 = PCA(n_components=_var373)
train_reduced_data = pca_2.fit_transform(X_train)
_var374 = 'hinge'
_var375 = 'elasticnet'
_var376 = 0.001
_var377 = 0.6
_var378 = 10000
clf_2 = SGDClassifier(loss=_var374, penalty=_var375, alpha=_var376, l1_ratio=_var377, n_iter=_var378)
clf_3 = clf_2.fit(train_reduced_data, y_train)
test_reduced_data = pca_2.transform(X_test)
pred_0 = clf_3.predict(test_reduced_data)
_var379 = accuracy_score(y_test, pred_0)
_var380 = 3
score_accuracy_0 = round(_var379, _var380)
_var381 = f1_score(y_test, pred_0)
_var382 = 3
score_f1_0 = round(_var381, _var382)
_var383 = precision_score(y_test, pred_0)
_var384 = 3
score_precision = round(_var383, _var384)
_var385 = recall_score(y_test, pred_0)
_var386 = 3
score_recall = round(_var385, _var386)
(score_accuracy_0, score_f1_0, score_precision, score_recall)
