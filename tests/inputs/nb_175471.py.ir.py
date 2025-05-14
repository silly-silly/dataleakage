

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
import time
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold, KFold
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import random
from sklearn import preprocessing
from sklearn.naive_bayes import MultinomialNB
import gc
from scipy.stats import skew, boxcox
from bayes_opt import BayesianOptimization
from scipy import sparse
from sklearn.metrics import log_loss
from datetime import datetime
from scipy.stats.mstats import gmean
import matplotlib.pyplot as plt
import seaborn as sns
_var0 = get_ipython()
_var1 = 'matplotlib'
_var2 = 'inline'
_var0.run_line_magic(_var1, _var2)
seed = 2017
data_path = '../input/'
_var3 = 'train_BM_MB_add03052240.csv'
_var4 = (data_path + _var3)
train_X = pd.read_csv(_var4)
_var5 = 'test_BM_MB_add03052240.csv'
_var6 = (data_path + _var5)
test_X = pd.read_csv(_var6)
_var7 = 'labels_BrandenMurray.csv'
_var8 = (data_path + _var7)
_var9 = pd.read_csv(_var8)
train_y = np.ravel(_var9)
_var10 = train_X.shape
_var11 = 0
ntrain = _var10[_var11]
_var12 = test_X.listing_id
_var13 = _var12.values
sub_list = _var13.copy()
_var14 = train_X.shape
_var15 = test_X.shape
_var16 = train_y.shape
_var17 = (_var14, _var15, _var16)
print(_var17)
_var18 = [train_X, test_X]
full_data = pd.concat(_var18)
_var19 = train_X.columns
features_to_use = _var19.values
_var20 = full_data[features_to_use]

def _func0(x):
    _var21 = x.dropna()
    _var22 = skew(_var21)
    return _var22
skewed_cols = _var20.apply(_func0)
SSL = preprocessing.StandardScaler()
_var23 = 0.25
_var24 = (skewed_cols > _var23)
_var25 = skewed_cols[_var24]
_var26 = _var25.index
skewed_cols_0 = _var26.values
for skewed_col in skewed_cols_0:
    _var29 = full_data[skewed_col]
    _var30 = full_data[skewed_col]
    _var31 = _var30.min()
    _var32 = (_var29 - _var31)
    _var33 = 1
    _var34 = (_var32 + _var33)
    (_var27, _var28) = boxcox(_var34)
    full_data_0 = set_index_wrapper(full_data, skewed_col, _var27)
    lam = _var28
full_data_1 = __phi__(full_data_0, full_data)
for col in features_to_use:
    _var35 = full_data_1[col]
    _var36 = _var35.values
    _var37 = (- 1)
    _var38 = 1
    _var39 = _var36.reshape(_var37, _var38)
    _var40 = SSL.fit_transform(_var39)
    full_data_2 = set_index_wrapper(full_data_1, col, _var40)
    _var41 = full_data_2[col]
    _var42 = full_data_2[col]
    _var43 = _var42.min()
    _var44 = (_var41 - _var43)
    _var45 = 1
    _var46 = (_var44 + _var45)
    full_data_3 = set_index_wrapper(full_data_2, col, _var46)
    _var47 = full_data_3.iloc
    _var48 = _var47[:ntrain]
    _var49 = _var48[col]
    train_X_0 = set_index_wrapper(train_X, col, _var49)
    _var50 = full_data_3.iloc
    _var51 = _var50[ntrain:]
    _var52 = _var51[col]
    test_X_0 = set_index_wrapper(test_X, col, _var52)
full_data_4 = __phi__(full_data_3, full_data_1)
train_X_1 = __phi__(train_X_0, train_X)
test_X_1 = __phi__(test_X_0, test_X)
del full_data_4
_var57 = 0.2
(_var53, _var54, _var55, _var56) = train_test_split(train_X_1, train_y, test_size=_var57, random_state=seed)
X_train = _var53
X_val = _var54
y_train = _var55
y_val = _var56

def MNB_cv(alpha=1.0):
    scores = []
    est = MultinomialNB(alpha=alpha)
    _var58 = global_wrapper(X_train)
    _var59 = global_wrapper(y_train)
    est_0 = est.fit(_var58, _var59)
    _var60 = global_wrapper(X_val)
    y_val_pred = est_0.predict_proba(_var60)
    _var61 = (- 1)
    _var62 = global_wrapper(y_val)
    _var63 = log_loss(_var62, y_val_pred)
    _var64 = (_var61 * _var63)
    return _var64
cv_score = (- 1)
_var65 = 600
_var66 = 1000
_var67 = 50
_var68 = range(_var65, _var66, _var67)
for x_0 in _var68:
    score = MNB_cv(alpha=x_0)
    _var69 = (score > cv_score)
    if _var69:
        alpha_0 = x_0
        cv_score_0 = score
    cv_score_1 = __phi__(cv_score_0, cv_score)
    _var70 = 'alpha = {0}\t {1:.12}'
    _var71 = _var70.format(x_0, score)
    print(_var71)
cv_score_2 = __phi__(cv_score_1, cv_score)

def MNB_blend(est_1, train_x, train_y_0, test_x, fold):
    N_params = len(est_1)
    _var72 = 'Blend %d estimators for %d folds'
    _var73 = (N_params, fold)
    _var74 = (_var72 % _var73)
    print(_var74)
    _var75 = global_wrapper(seed)
    skf = KFold(n_splits=fold, random_state=_var75)
    _var76 = set(train_y_0)
    N_class = len(_var76)
    _var77 = train_x.shape
    _var78 = 0
    _var79 = _var77[_var78]
    _var80 = (N_class * N_params)
    _var81 = (_var79, _var80)
    train_blend_x = np.zeros(_var81)
    _var82 = test_x.shape
    _var83 = 0
    _var84 = _var82[_var83]
    _var85 = (N_class * N_params)
    _var86 = (_var84, _var85)
    test_blend_x_mean = np.zeros(_var86)
    _var87 = test_x.shape
    _var88 = 0
    _var89 = _var87[_var88]
    _var90 = (N_class * N_params)
    _var91 = (_var89, _var90)
    test_blend_x_gmean = np.zeros(_var91)
    _var92 = (fold, N_params)
    scores_0 = np.zeros(_var92)
    _var93 = (fold, N_params)
    best_rounds = np.zeros(_var93)
    _var94 = enumerate(est_1)
    for _var95 in _var94:
        _var98 = 0
        j = _var95[_var98]
        _var99 = 1
        ester = _var95[_var99]
        _var100 = 'Model %d:'
        _var101 = 1
        _var102 = (j + _var101)
        _var103 = (_var100 % _var102)
        print(_var103)
        _var104 = test_x.shape
        _var105 = 0
        _var106 = _var104[_var105]
        _var107 = (N_class * fold)
        _var108 = (_var106, _var107)
        test_blend_x_j = np.zeros(_var108)
        _var109 = skf.split(train_x)
        _var110 = enumerate(_var109)
        for _var111 in _var110:
            _var114 = 0
            i = _var111[_var114]
            _var117 = 1
            _var118 = _var111[_var117]
            _var119 = 0
            train_index = _var118[_var119]
            _var120 = 1
            val_index = _var118[_var120]
            _var121 = 'Model %d fold %d'
            _var122 = 1
            _var123 = (j + _var122)
            _var124 = 1
            _var125 = (i + _var124)
            _var126 = (_var123, _var125)
            _var127 = (_var121 % _var126)
            print(_var127)
            fold_start = time.time()
            _var128 = train_x.iloc
            train_x_fold = _var128[train_index]
            train_y_fold = train_y_0[train_index]
            _var129 = train_x.iloc
            val_x_fold = _var129[val_index]
            val_y_fold = train_y_0[val_index]
            ester_0 = ester.fit(train_x_fold, train_y_fold)
            val_y_predict_fold = ester_0.predict_proba(val_x_fold)
            score_0 = log_loss(val_y_fold, val_y_predict_fold)
            _var130 = 'Score: '
            _var131 = (_var130, score_0)
            print(_var131)
            _var132 = (i, j)
            scores_1 = set_index_wrapper(scores_0, _var132, score_0)
            _var133 = (j * N_class)
            _var134 = 1
            _var135 = (j + _var134)
            _var136 = (_var135 * N_class)
            train_blend_x_0 = set_index_wrapper(train_blend_x, (val_index, slice(_var133, _var136, None)), val_y_predict_fold)
            _var137 = (i * N_class)
            _var138 = 1
            _var139 = (i + _var138)
            _var140 = (_var139 * N_class)
            _var141 = ester_0.predict_proba(test_x)
            test_blend_x_j_0 = set_index_wrapper(test_blend_x_j, (slice(None, None, None), slice(_var137, _var140, None)), _var141)
            _var142 = 'Model %d fold %d fitting finished in %0.3fs'
            _var143 = 1
            _var144 = (j + _var143)
            _var145 = 1
            _var146 = (i + _var145)
            _var147 = time.time()
            _var148 = (_var147 - fold_start)
            _var149 = (_var144, _var146, _var148)
            _var150 = (_var142 % _var149)
            print(_var150)
        ester_1 = __phi__(ester_0, ester)
        train_blend_x_1 = __phi__(train_blend_x_0, train_blend_x)
        test_blend_x_j_1 = __phi__(test_blend_x_j_0, test_blend_x_j)
        scores_2 = __phi__(scores_1, scores_0)
        score_1 = __phi__(score_0, score)
        _var151 = (j * N_class)
        _var152 = 1
        _var153 = (j + _var152)
        _var154 = (_var153 * N_class)
        _var155 = 0
        _var156 = (N_class * fold)
        _var157 = range(_var155, _var156, N_class)
        _var158 = list(_var157)
        _var159 = test_blend_x_j_1[:, _var158]
        _var160 = 1
        _var161 = _var159.mean(_var160)
        _var162 = 1
        _var163 = (N_class * fold)
        _var164 = range(_var162, _var163, N_class)
        _var165 = list(_var164)
        _var166 = test_blend_x_j_1[:, _var165]
        _var167 = 1
        _var168 = _var166.mean(_var167)
        _var169 = 2
        _var170 = (N_class * fold)
        _var171 = range(_var169, _var170, N_class)
        _var172 = list(_var171)
        _var173 = test_blend_x_j_1[:, _var172]
        _var174 = 1
        _var175 = _var173.mean(_var174)
        _var176 = [_var161, _var168, _var175]
        _var177 = np.stack(_var176)
        _var178 = _var177.T
        test_blend_x_mean_0 = set_index_wrapper(test_blend_x_mean, (slice(None, None, None), slice(_var151, _var154, None)), _var178)
        _var179 = (j * N_class)
        _var180 = 1
        _var181 = (j + _var180)
        _var182 = (_var181 * N_class)
        _var183 = 0
        _var184 = (N_class * fold)
        _var185 = range(_var183, _var184, N_class)
        _var186 = list(_var185)
        _var187 = test_blend_x_j_1[:, _var186]
        _var188 = 1
        _var189 = gmean(_var187, axis=_var188)
        _var190 = 1
        _var191 = (N_class * fold)
        _var192 = range(_var190, _var191, N_class)
        _var193 = list(_var192)
        _var194 = test_blend_x_j_1[:, _var193]
        _var195 = 1
        _var196 = gmean(_var194, axis=_var195)
        _var197 = 2
        _var198 = (N_class * fold)
        _var199 = range(_var197, _var198, N_class)
        _var200 = list(_var199)
        _var201 = test_blend_x_j_1[:, _var200]
        _var202 = 1
        _var203 = gmean(_var201, axis=_var202)
        _var204 = [_var189, _var196, _var203]
        _var205 = np.stack(_var204)
        _var206 = _var205.T
        test_blend_x_gmean_0 = set_index_wrapper(test_blend_x_gmean, (slice(None, None, None), slice(_var179, _var182, None)), _var206)
        _var207 = 'Score for model %d is %f'
        _var208 = 1
        _var209 = (j + _var208)
        _var210 = scores_2[:, j]
        _var211 = np.mean(_var210)
        _var212 = (_var209, _var211)
        _var213 = (_var207 % _var212)
        print(_var213)
    train_blend_x_2 = __phi__(train_blend_x_1, train_blend_x)
    test_blend_x_mean_1 = __phi__(test_blend_x_mean_0, test_blend_x_mean)
    scores_3 = __phi__(scores_2, scores_0)
    score_2 = __phi__(score_1, score)
    test_blend_x_gmean_1 = __phi__(test_blend_x_gmean_0, test_blend_x_gmean)
    _var214 = 'Score for blended models is %f'
    _var215 = np.mean(scores_3)
    _var216 = (_var214 % _var215)
    print(_var216)
    return (train_blend_x_2, test_blend_x_mean_1, test_blend_x_gmean_1, scores_3, best_rounds)
_var217 = 800
_var218 = MultinomialNB(alpha=_var217)
_var219 = 850
_var220 = MultinomialNB(alpha=_var219)
_var221 = 900
_var222 = MultinomialNB(alpha=_var221)
est_2 = [_var218, _var220, _var222]
_var228 = 10
(_var223, _var224, _var225, _var226, _var227) = MNB_blend(est_2, train_X_1, train_y, test_X_1, _var228)
train_blend_x_MNB = _var223
test_blend_x_MNB_mean = _var224
test_blend_x_MNB_gmean = _var225
blend_scores_MNB = _var226
best_rounds_MNB = _var227
now = datetime.now()
_var229 = '../blend/train_blend_MNB_BM_MB_add03052240_'
_var230 = '%Y-%m-%d-%H-%M'
_var231 = now.strftime(_var230)
_var232 = str(_var231)
_var233 = (_var229 + _var232)
_var234 = '.csv'
name_train_blend = (_var233 + _var234)
_var235 = '../blend/test_blend_MNB_mean_BM_MB_add03052240_'
_var236 = '%Y-%m-%d-%H-%M'
_var237 = now.strftime(_var236)
_var238 = str(_var237)
_var239 = (_var235 + _var238)
_var240 = '.csv'
name_test_blend_mean = (_var239 + _var240)
_var241 = '../blend/test_blend_MNB_gmean_BM_MB_add03052240_'
_var242 = '%Y-%m-%d-%H-%M'
_var243 = now.strftime(_var242)
_var244 = str(_var243)
_var245 = (_var241 + _var244)
_var246 = '.csv'
name_test_blend_gmean = (_var245 + _var246)
_var247 = 0
_var248 = np.mean(blend_scores_MNB, axis=_var247)
print(_var248)
_var249 = ','
np.savetxt(name_train_blend, train_blend_x_MNB, delimiter=_var249)
_var250 = ','
np.savetxt(name_test_blend_mean, test_blend_x_MNB_mean, delimiter=_var250)
_var251 = ','
np.savetxt(name_test_blend_gmean, test_blend_x_MNB_gmean, delimiter=_var251)
_var252 = '../output/sub_MNB_mean_BM_MB_add03052240_'
_var253 = '%Y-%m-%d-%H-%M'
_var254 = now.strftime(_var253)
_var255 = str(_var254)
_var256 = (_var252 + _var255)
_var257 = '.csv'
sub_name = (_var256 + _var257)
_var258 = 3
_var259 = test_blend_x_MNB_mean[:, :_var258]
out_df = pd.DataFrame(_var259)
_var260 = ['low', 'medium', 'high']
out_df_0 = set_field_wrapper(out_df, 'columns', _var260)
_var261 = 'listing_id'
out_df_1 = set_index_wrapper(out_df_0, _var261, sub_list)
_var262 = False
out_df_1.to_csv(sub_name, index=_var262)
