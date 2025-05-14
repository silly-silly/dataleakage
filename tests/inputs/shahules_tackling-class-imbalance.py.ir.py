

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
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, TruncatedSVD
import time
import matplotlib.patches as mpatches
from sklearn.metrics import confusion_matrix
_var0 = get_ipython()
_var1 = 'time'
_var2 = ''
_var3 = "train_transactions=pd.read_csv('../input/train_transaction.csv')\ntrain_identity=pd.read_csv('../input/train_identity.csv')\nprint('Train data set is loaded !')"
_var0.run_cell_magic(_var1, _var2, _var3)
_var4 = global_wrapper(train_transactions)
_var4.head()
_var5 = global_wrapper(train_transactions)
_var5.info()
_var6 = global_wrapper(train_identity)
_var6.info()
_var7 = global_wrapper(train_transactions)
_var8 = 'isFraud'
_var9 = _var7[_var8]
_var10 = _var9.value_counts()
x = _var10.values
_var11 = [0, 1]
sns.barplot(_var11, x)
_var12 = 'Target variable count'
plt.title(_var12)
_var13 = global_wrapper(train_transactions)
_var14 = global_wrapper(train_identity)
_var15 = 'left'
_var16 = True
_var17 = True
train = _var13.merge(_var14, how=_var15, left_index=_var16, right_index=_var17)
_var18 = 'isFraud'
_var19 = train[_var18]
_var20 = 'uint8'
y_train = _var19.astype(_var20)
_var21 = 'Train shape'
_var22 = train.shape
_var23 = (_var21, _var22)
print(_var23)
_var24 = global_wrapper(train_transactions)
_var25 = global_wrapper(train_identity)
del _var24, _var25
_var26 = 'Data set merged '
print(_var26)
_var27 = get_ipython()
_var28 = 'time'
_var29 = ''
_var30 = '# From kernel https://www.kaggle.com/gemartin/load-data-reduce-memory-usage\n# WARNING! THIS CAN DAMAGE THE DATA \ndef reduce_mem_usage2(df):\n    """ iterate through all the columns of a dataframe and modify the data type\n        to reduce memory usage.        \n    """\n    start_mem = df.memory_usage().sum() / 1024**2\n    print(\'Memory usage of dataframe is {:.2f} MB\'.format(start_mem))\n    \n    for col in df.columns:\n        col_type = df[col].dtype\n        \n        if col_type != object:\n            c_min = df[col].min()\n            c_max = df[col].max()\n            if str(col_type)[:3] == \'int\':\n                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:\n                    df[col] = df[col].astype(np.int8)\n                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:\n                    df[col] = df[col].astype(np.int16)\n                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:\n                    df[col] = df[col].astype(np.int32)\n                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:\n                    df[col] = df[col].astype(np.int64)  \n            else:\n                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:\n                    df[col] = df[col].astype(np.float16)\n                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:\n                    df[col] = df[col].astype(np.float32)\n                else:\n                    df[col] = df[col].astype(np.float64)\n        else:\n            df[col] = df[col].astype(\'category\')\n\n    end_mem = df.memory_usage().sum() / 1024**2\n    print(\'Memory usage after optimization is: {:.2f} MB\'.format(end_mem))\n    print(\'Decreased by {:.1f}%\'.format(100 * (start_mem - end_mem) / start_mem))\n    \n    return df\n'
_var27.run_cell_magic(_var28, _var29, _var30)
_var31 = get_ipython()
_var32 = 'time'
_var33 = ''
_var34 = 'train = reduce_mem_usage2(train)\n\n'
_var31.run_cell_magic(_var32, _var33, _var34)
_var39 = 'isFraud'
_var40 = 1
_var41 = train.drop(_var39, axis=_var40)
_var42 = 0.2
_var43 = 1
(_var35, _var36, _var37, _var38) = train_test_split(_var41, y_train, test_size=_var42, random_state=_var43)
X_train = _var35
X_test = _var36
y_train_0 = _var37
y_test = _var38
_var44 = [X_train, y_train_0]
_var45 = 1
X = pd.concat(_var44, axis=_var45)
_var46 = X.isFraud
_var47 = 0
_var48 = (_var46 == _var47)
not_fraud = X[_var48]
_var49 = X.isFraud
_var50 = 1
_var51 = (_var49 == _var50)
fraud = X[_var51]
_var52 = True
_var53 = len(not_fraud)
_var54 = 27
fraud_upsampled = resample(fraud, replace=_var52, n_samples=_var53, random_state=_var54)
_var55 = [not_fraud, fraud_upsampled]
upsampled = pd.concat(_var55)
_var56 = upsampled.isFraud
_var56.value_counts()
_var57 = upsampled.isFraud
y = _var57.value_counts()
_var58 = [0, 1]
sns.barplot(y=y, x=_var58)
_var59 = 'upsampled data class count'
plt.title(_var59)
_var60 = 'count'
plt.ylabel(_var60)
_var61 = False
_var62 = len(fraud)
_var63 = 27
not_fraud_downsampled = resample(not_fraud, replace=_var61, n_samples=_var62, random_state=_var63)
_var64 = [not_fraud_downsampled, fraud]
downsampled = pd.concat(_var64)
_var65 = downsampled.isFraud
_var65.value_counts()
_var66 = downsampled.isFraud
y_0 = _var66.value_counts()
_var67 = [0, 1]
sns.barplot(y=y_0, x=_var67)
_var68 = 'downsampled data class count'
plt.title(_var68)
_var69 = 'count'
plt.ylabel(_var69)
from sklearn.datasets import make_classification
_var72 = 2
_var73 = 1.5
_var74 = [0.9, 0.1]
_var75 = 3
_var76 = 1
_var77 = 0
_var78 = 20
_var79 = 1
_var80 = 1000
_var81 = 10
(_var70, _var71) = make_classification(n_classes=_var72, class_sep=_var73, weights=_var74, n_informative=_var75, n_redundant=_var76, flip_y=_var77, n_features=_var78, n_clusters_per_class=_var79, n_samples=_var80, random_state=_var81)
X_0 = _var70
y_1 = _var71
df = pd.DataFrame(X_0)
_var82 = 'target'
df_0 = set_index_wrapper(df, _var82, y_1)
_var83 = df_0.target
_var84 = _var83.value_counts()
_var85 = 'bar'
_var86 = 'Count (target)'
_var84.plot(kind=_var85, title=_var86)

def logistic(X_1, y_2):
    _var91 = 0.2
    _var92 = 1
    (_var87, _var88, _var89, _var90) = train_test_split(X_1, y_2, test_size=_var91, random_state=_var92)
    X_train_0 = _var87
    X_test_0 = _var88
    y_train_1 = _var89
    y_test_0 = _var90
    lr = LogisticRegression()
    lr_0 = lr.fit(X_train_0, y_train_1)
    prob = lr_0.predict_proba(X_test_0)
    _var93 = 1
    _var94 = prob[:, _var93]
    return (_var94, y_test_0)
(_var95, _var96) = logistic(X_0, y_1)
probs = _var95
y_test_1 = _var96

def plot_pre_curve(y_test_2, probs_0):
    (_var97, _var98, _var99) = precision_recall_curve(y_test_2, probs_0)
    precision = _var97
    recall = _var98
    thresholds = _var99
    _var100 = [0, 1]
    _var101 = [0.5, 0.5]
    _var102 = '--'
    plt.plot(_var100, _var101, linestyle=_var102)
    _var103 = '.'
    plt.plot(recall, precision, marker=_var103)
    _var104 = 'precision recall curve'
    plt.title(_var104)
    _var105 = 'Recall'
    plt.xlabel(_var105)
    _var106 = 'Precision'
    plt.ylabel(_var106)
    plt.show()

def plot_roc(y_test_3, prob_0):
    _var110 = global_wrapper(probs)
    (_var107, _var108, _var109) = roc_curve(y_test_3, _var110)
    fpr = _var107
    tpr = _var108
    thresholds_0 = _var109
    _var111 = [0, 1]
    _var112 = [0, 1]
    _var113 = '--'
    plt.plot(_var111, _var112, linestyle=_var113)
    _var114 = '.'
    plt.plot(fpr, tpr, marker=_var114)
    _var115 = 'ROC curve'
    plt.title(_var115)
    _var116 = 'false positive rate'
    plt.xlabel(_var116)
    _var117 = 'true positive rate'
    plt.ylabel(_var117)
    plt.show()
plot_pre_curve(y_test_1, probs)
plot_roc(y_test_1, probs)

def plot_2d_space(X_train_1, y_train_2, X_2=X, y_3=y, label='Classes'):
    colors = ['#1F77B4', '#FF7F0E']
    markers = ['o', 's']
    _var120 = 1
    _var121 = 2
    _var122 = 8
    _var123 = 4
    _var124 = (_var122, _var123)
    (_var118, _var119) = plt.subplots(_var120, _var121, figsize=_var124)
    fig = _var118
    _var127 = 0
    ax1 = _var119[_var127]
    _var128 = 1
    ax2 = _var119[_var128]
    _var129 = np.unique(y_3)
    _var130 = zip(_var129, colors, markers)
    for _var131 in _var130:
        _var135 = 0
        l = _var131[_var135]
        _var136 = 1
        c = _var131[_var136]
        _var137 = 2
        m = _var131[_var137]
        _var138 = global_wrapper(ax1)
        _var139 = (y_train_2 == l)
        _var140 = 0
        _var141 = (_var139, _var140)
        _var142 = X_train_1[_var141]
        _var143 = (y_train_2 == l)
        _var144 = 1
        _var145 = (_var143, _var144)
        _var146 = X_train_1[_var145]
        _var138.scatter(_var142, _var146, c=c, label=l, marker=m)
    _var147 = np.unique(y_3)
    _var148 = zip(_var147, colors, markers)
    for _var149 in _var148:
        _var153 = 0
        l_0 = _var149[_var153]
        _var154 = 1
        c_0 = _var149[_var154]
        _var155 = 2
        m_0 = _var149[_var155]
        _var156 = global_wrapper(ax2)
        _var157 = (y_3 == l_0)
        _var158 = 0
        _var159 = (_var157, _var158)
        _var160 = X_2[_var159]
        _var161 = (y_3 == l_0)
        _var162 = 1
        _var163 = (_var161, _var162)
        _var164 = X_2[_var163]
        _var156.scatter(_var160, _var164, c=c_0, label=l_0, marker=m_0)
    c_1 = __phi__(c_0, c)
    l_1 = __phi__(l_0, l)
    m_1 = __phi__(m_0, m)
    _var165 = global_wrapper(ax1)
    _var165.set_title(label)
    _var166 = global_wrapper(ax2)
    _var167 = 'original data'
    _var166.set_title(_var167)
    _var168 = 'upper right'
    plt.legend(loc=_var168)
    plt.show()
t0 = time.time()
_var169 = 2
_var170 = 42
_var171 = TSNE(n_components=_var169, random_state=_var170)
X_reduced_tsne = _var171.fit_transform(X_0)
t1 = time.time()
_var172 = 'T-SNE took {:.2} s'
_var173 = (t1 - t0)
_var174 = _var172.format(_var173)
print(_var174)
t0_0 = time.time()
_var175 = 2
_var176 = 42
_var177 = PCA(n_components=_var175, random_state=_var176)
X_reduced_pca = _var177.fit_transform(X_0)
t1_0 = time.time()
_var178 = 'PCA took {:.2} s'
_var179 = (t1_0 - t0_0)
_var180 = _var178.format(_var179)
print(_var180)
t0_1 = time.time()
_var181 = 2
_var182 = 'randomized'
_var183 = 42
_var184 = TruncatedSVD(n_components=_var181, algorithm=_var182, random_state=_var183)
X_reduced_svd = _var184.fit_transform(X_0)
t1_1 = time.time()
_var185 = 'Truncated SVD took {:.2} s'
_var186 = (t1_1 - t0_1)
_var187 = _var185.format(_var186)
print(_var187)
_var190 = 1
_var191 = 3
_var192 = 24
_var193 = 6
_var194 = (_var192, _var193)
(_var188, _var189) = plt.subplots(_var190, _var191, figsize=_var194)
f = _var188
_var198 = 0
ax1 = _var189[_var198]
_var199 = 1
ax2 = _var189[_var199]
_var200 = 2
ax3 = _var189[_var200]
_var201 = 'Clusters using Dimensionality Reduction'
_var202 = 14
f.suptitle(_var201, fontsize=_var202)
_var203 = '#0A0AFF'
_var204 = 'No Fraud'
blue_patch = mpatches.Patch(color=_var203, label=_var204)
_var205 = '#AF0000'
_var206 = 'Fraud'
red_patch = mpatches.Patch(color=_var205, label=_var206)
_var207 = global_wrapper(ax1)
_var208 = 0
_var209 = X_reduced_tsne[:, _var208]
_var210 = 1
_var211 = X_reduced_tsne[:, _var210]
_var212 = 0
_var213 = (y_1 == _var212)
_var214 = 'coolwarm'
_var215 = 'No Fraud'
_var216 = 2
_var207.scatter(_var209, _var211, c=_var213, cmap=_var214, label=_var215, linewidths=_var216)
_var217 = global_wrapper(ax1)
_var218 = 0
_var219 = X_reduced_tsne[:, _var218]
_var220 = 1
_var221 = X_reduced_tsne[:, _var220]
_var222 = 1
_var223 = (y_1 == _var222)
_var224 = 'coolwarm'
_var225 = 'Fraud'
_var226 = 2
_var217.scatter(_var219, _var221, c=_var223, cmap=_var224, label=_var225, linewidths=_var226)
_var227 = global_wrapper(ax1)
_var228 = 't-SNE'
_var229 = 14
_var227.set_title(_var228, fontsize=_var229)
_var230 = global_wrapper(ax1)
_var231 = True
_var230.grid(_var231)
_var232 = global_wrapper(ax1)
_var233 = [blue_patch, red_patch]
_var232.legend(handles=_var233)
_var234 = global_wrapper(ax2)
_var235 = 0
_var236 = X_reduced_pca[:, _var235]
_var237 = 1
_var238 = X_reduced_pca[:, _var237]
_var239 = 0
_var240 = (y_1 == _var239)
_var241 = 'coolwarm'
_var242 = 'No Fraud'
_var243 = 2
_var234.scatter(_var236, _var238, c=_var240, cmap=_var241, label=_var242, linewidths=_var243)
_var244 = global_wrapper(ax2)
_var245 = 0
_var246 = X_reduced_pca[:, _var245]
_var247 = 1
_var248 = X_reduced_pca[:, _var247]
_var249 = 1
_var250 = (y_1 == _var249)
_var251 = 'coolwarm'
_var252 = 'Fraud'
_var253 = 2
_var244.scatter(_var246, _var248, c=_var250, cmap=_var251, label=_var252, linewidths=_var253)
_var254 = global_wrapper(ax2)
_var255 = 'PCA'
_var256 = 14
_var254.set_title(_var255, fontsize=_var256)
_var257 = global_wrapper(ax2)
_var258 = True
_var257.grid(_var258)
_var259 = global_wrapper(ax2)
_var260 = [blue_patch, red_patch]
_var259.legend(handles=_var260)
_var261 = global_wrapper(ax3)
_var262 = 0
_var263 = X_reduced_svd[:, _var262]
_var264 = 1
_var265 = X_reduced_svd[:, _var264]
_var266 = 0
_var267 = (y_1 == _var266)
_var268 = 'coolwarm'
_var269 = 'No Fraud'
_var270 = 2
_var261.scatter(_var263, _var265, c=_var267, cmap=_var268, label=_var269, linewidths=_var270)
_var271 = global_wrapper(ax3)
_var272 = 0
_var273 = X_reduced_svd[:, _var272]
_var274 = 1
_var275 = X_reduced_svd[:, _var274]
_var276 = 1
_var277 = (y_1 == _var276)
_var278 = 'coolwarm'
_var279 = 'Fraud'
_var280 = 2
_var271.scatter(_var273, _var275, c=_var277, cmap=_var278, label=_var279, linewidths=_var280)
_var281 = global_wrapper(ax3)
_var282 = 'Truncated SVD'
_var283 = 14
_var281.set_title(_var282, fontsize=_var283)
_var284 = global_wrapper(ax3)
_var285 = True
_var284.grid(_var285)
_var286 = global_wrapper(ax3)
_var287 = [blue_patch, red_patch]
_var286.legend(handles=_var287)
plt.show()
import imblearn
from imblearn.under_sampling import RandomUnderSampler
_var288 = True
ran = RandomUnderSampler(return_indices=_var288)
(_var289, _var290, _var291) = ran.fit_sample(X_0, y_1)
X_rs = _var289
y_rs = _var290
dropped = _var291
_var292 = 'The number of removed indices are '
_var293 = len(dropped)
_var294 = (_var292, _var293)
print(_var294)
_var295 = 'Random under sampling'
plot_2d_space(X_rs, y_rs, X_0, y_1, _var295)
(_var296, _var297) = logistic(X_rs, y_rs)
probs_1 = _var296
y_test_4 = _var297
plot_pre_curve(y_test_4, probs_1)
plot_roc(y_test_4, probs_1)
from imblearn.over_sampling import RandomOverSampler
ran_0 = RandomOverSampler()
(_var298, _var299) = ran_0.fit_resample(X_0, y_1)
X_ran = _var298
y_ran = _var299
_var300 = 'The new data contains {} rows '
_var301 = X_ran.shape
_var302 = 0
_var303 = _var301[_var302]
_var304 = _var300.format(_var303)
print(_var304)
_var305 = 'over-sampled'
plot_2d_space(X_ran, y_ran, X_0, y_1, _var305)
(_var306, _var307) = logistic(X_ran, y_ran)
probs_2 = _var306
y_test_5 = _var307
plot_pre_curve(y_test_5, probs_2)
plot_roc(y_test_5, probs_2)
from imblearn.under_sampling import TomekLinks
_var308 = True
_var309 = 'majority'
tl = TomekLinks(return_indices=_var308, ratio=_var309)
(_var310, _var311, _var312) = tl.fit_sample(X_0, y_1)
X_tl = _var310
y_tl = _var311
id_tl = _var312
_var313 = 'Tomek links under-sampling'
plot_2d_space(X_tl, y_tl, X_0, y_1, _var313)
(_var314, _var315) = logistic(X_tl, y_tl)
probs_3 = _var314
y_test_6 = _var315
plot_pre_curve(y_test_6, probs_3)
plot_roc(y_test_6, probs_3)
from imblearn.over_sampling import SMOTE
_var316 = 'minority'
smote = SMOTE(ratio=_var316)
(_var317, _var318) = smote.fit_sample(X_0, y_1)
X_sm = _var317
y_sm = _var318
_var319 = 'SMOTE over-sampling'
plot_2d_space(X_sm, y_sm, X_0, y_1, _var319)
(_var320, _var321) = logistic(X_sm, y_sm)
probs_4 = _var320
y_test_7 = _var321
plot_pre_curve(y_test_7, probs_4)
plot_roc(y_test_7, probs_4)
