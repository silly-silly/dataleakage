

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
import sys
import pandas as pd
import numpy as np
_var3 = 'display.max_columns'
_var4 = None
pd.set_option(_var3, _var4)
import sklearn.preprocessing as preprocessing
import sklearn.feature_extraction as feature_extraction
from sklearn.metrics import confusion_matrix, make_scorer, recall_score, classification_report, roc_curve, auc, roc_auc_score
from sklearn.linear_model import SGDClassifier, PassiveAggressiveClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, ParameterSampler
from sklearn.kernel_approximation import RBFSampler
from scipy.stats.distributions import expon
from sklearn.feature_selection import SelectKBest, chi2, GenericUnivariateSelect
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, ShuffleSplit, StratifiedShuffleSplit
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.neural_network import BernoulliRBM
import seaborn as sns
sns.set()
import matplotlib
from operator import itemgetter
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from scipy.stats import randint as sp_randint
_var5 = matplotlib.rcParams
_var6 = 'figure.figsize'
_var7 = 13.0
_var8 = 13.0
_var9 = (_var7, _var8)
_var5_0 = set_index_wrapper(_var5, _var6, _var9)
import time

class ItemSelector(BaseEstimator, TransformerMixin):
    '\n    Parameters\n    ----------\n    key : hashable, required\n        The key corresponding to the desired value in a mappable.\n    '

    def __init__(self, key):
        self_0 = set_field_wrapper(self, 'key', key)

    def fit(self_1, x, y=None):
        return self_1

    def transform(self_2, data_dict):
        _var10 = self_2.key
        _var11 = data_dict[_var10]
        return _var11

    def inverse_transform(self_3, X):
        return X

def plot_learning_curve(estimator, title, X_0, y_0, ylim=None, cv=None, n_jobs=1, train_sizes=np.linspace(0.1, 1.0, 5)):
    '\n    Generate a simple plot of the test and traning learning curve.\n\n    Parameters\n    ----------\n    estimator : object type that implements the "fit" and "predict" methods\n        An object of that type which is cloned for each validation.\n\n    title : string\n        Title for the chart.\n\n    X : array-like, shape (n_samples, n_features)\n        Training vector, where n_samples is the number of samples and\n        n_features is the number of features.\n\n    y : array-like, shape (n_samples) or (n_samples, n_features), optional\n        Target relative to X for classification or regression;\n        None for unsupervised learning.\n\n    ylim : tuple, shape (ymin, ymax), optional\n        Defines minimum and maximum yvalues plotted.\n\n    cv : integer, cross-validation generator, optional\n        If an integer is passed, it is the number of folds (defaults to 3).\n        Specific cross-validation objects can be passed, see\n        sklearn.cross_validation module for the list of possible objects\n\n    n_jobs : integer, optional\n        Number of jobs to run in parallel (default 1).\n    '
    plt.figure()
    plt.title(title)
    _var12 = None
    _var13 = (ylim is not _var12)
    if _var13:
        plt.ylim(*ylim)
    _var14 = 'Training examples'
    plt.xlabel(_var14)
    _var15 = 'Score'
    plt.ylabel(_var15)
    (_var16, _var17, _var18) = learning_curve(estimator, X_0, y_0, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_sizes_0 = _var16
    train_scores = _var17
    test_scores = _var18
    _var19 = 1
    train_scores_mean = np.mean(train_scores, axis=_var19)
    _var20 = 1
    train_scores_std = np.std(train_scores, axis=_var20)
    _var21 = 1
    test_scores_mean = np.mean(test_scores, axis=_var21)
    _var22 = 1
    test_scores_std = np.std(test_scores, axis=_var22)
    plt.grid()
    _var23 = (train_scores_mean - train_scores_std)
    _var24 = (train_scores_mean + train_scores_std)
    _var25 = 0.1
    _var26 = 'r'
    plt.fill_between(train_sizes_0, _var23, _var24, alpha=_var25, color=_var26)
    _var27 = (test_scores_mean - test_scores_std)
    _var28 = (test_scores_mean + test_scores_std)
    _var29 = 0.1
    _var30 = 'g'
    plt.fill_between(train_sizes_0, _var27, _var28, alpha=_var29, color=_var30)
    _var31 = 'o-'
    _var32 = 'r'
    _var33 = 'Training score'
    plt.plot(train_sizes_0, train_scores_mean, _var31, color=_var32, label=_var33)
    _var34 = 'o-'
    _var35 = 'g'
    _var36 = 'Cross-validation score'
    plt.plot(train_sizes_0, test_scores_mean, _var34, color=_var35, label=_var36)
    _var37 = 'best'
    plt.legend(loc=_var37)
    return plt

def report(grid_scores, n_top=3):
    _var38 = 1
    _var39 = itemgetter(_var38)
    _var40 = True
    _var41 = sorted(grid_scores, key=_var39, reverse=_var40)
    top_scores = _var41[:n_top]
    _var42 = enumerate(top_scores)
    for _var43 in _var42:
        _var46 = 0
        i = _var43[_var46]
        _var47 = 1
        score = _var43[_var47]
        _var48 = 'Model with rank: {0}'
        _var49 = 1
        _var50 = (i + _var49)
        _var51 = _var48.format(_var50)
        print(_var51)
        _var52 = 'Mean validation score: {0:.3f} (std: {1:.3f})'
        _var53 = score.mean_validation_score
        _var54 = score.cv_validation_scores
        _var55 = np.std(_var54)
        _var56 = _var52.format(_var53, _var55)
        print(_var56)
        _var57 = 'Parameters: {0}'
        _var58 = score.parameters
        _var59 = _var57.format(_var58)
        print(_var59)
        _var60 = ''
        print(_var60)
_var61 = '/Users/felipelolas/PycharmProjects/FracasoEscolarChile/DatasetsProcesados/SIMCE/ALU/SIMCE_GEO_2013-2014.csv'
_var62 = 0
_var63 = '|'
_var64 = '.'
r = pd.read_csv(_var61, header=_var62, sep=_var63, decimal=_var64)
_var65 = r.columns
_var66 = [0, 29]
_var67 = _var65[_var66]
_var68 = 1
r_0 = r.drop(_var67, _var68)
_var69 = (- 999)
_var70 = 0
r_1 = r_0.fillna(_var69, axis=_var70)
cols = ['MRUN', 'COD_COM_ALU', 'NOM_COM_ALU', 'SIT_FIN_R', 'EDAD_ALU', 'CODINE11', 'LAT_MANZANA_ALU', 'LON_MANZANA_ALU', 'RIESGO_DESERCION_RBD', 'DIR_RBD', 'LAT_MANZANA_RBD', 'LON_MANZANA_RBD', 'CONVIVENCIA_2M_RBD', 'CONVIVENCIA_4B_RBD', 'CONVIVENCIA_6B_RBD', 'CONVIVENCIA_8B_RBD', 'AUTOESTIMA_MOTIVACION_2M_RBD', 'AUTOESTIMA_MOTIVACION_4B_RBD', 'AUTOESTIMA_MOTIVACION_6B_RBD', 'AUTOESTIMA_MOTIVACION_8B_RBD', 'PARTICIPACION_2M_RBD', 'PARTICIPACION_4B_RBD', 'PARTICIPACION_6B_RBD', 'PARTICIPACION_8B_RBD', 'IVE_MEDIA_RBD', 'IVE_BASICA_RBD', 'PSU_PROM_2013_RBD', 'CPAD_DISP', 'DGV_RBD', 'NOM_RBD', 'NOM_COM_RBD', 'COD_COM_RBD', 'RBD', 'PROM_GRAL', 'ASISTENCIA', 'LET_CUR', 'CLASIFICACION_SEP_RBD', 'MAT_TOTAL_RBD', 'MAT_MEDIA_RBD', 'MAT_BASICA_RBD', 'PROF_INSP_H_MAT_RBD', 'PROF_ORI_H_MAT_RBD', 'TIPO_ESTAB_MAT_RBD', 'POB_FLOT_RBD', 'VACANTES_CUR_IN_RBD', 'CANT_DOC_RBD']
_var71 = 1
data = r_1.drop(cols, _var71)
_var72 = 'IMPARTE'
_var73 = data.filter(like=_var72)
_var74 = _var73.columns
_var75 = 1
data_0 = data.drop(_var74, _var75)
_var76 = ['ABANDONA_ALU', 'DESERTA_ALU', 'ABANDONA_2014_ALU']
output_data = data_0[_var76]
_var77 = ['ABANDONA_ALU', 'DESERTA_ALU', 'ABANDONA_2014_ALU']
_var78 = 1
data_1 = data_0.drop(_var77, _var78)
ordinal_integer_cols = ['CANT_TRASLADOS_ALU', 'CANT_DELITOS_COM_ALU', 'CANT_CURSOS_RBD', 'CANT_DELITOS_MANZANA_RBD', 'CANT_DOC_M_RBD', 'CANT_DOC_F_RBD', 'PAGO_MATRICULA_RBD', 'PAGO_MENSUAL_RBD', 'SEL_SNED_RBD', 'BECAS_DISP_RBD', 'PROM_ALU_CUR_RBD', 'CANT_DELITOS_COM_RBD', 'EDU_P', 'EDU_M', 'ING_HOGAR']
_var79 = data_1.loc
_var80 = data_1.dtypes
_var81 = (_var80 == float)
ordinal_float_data = _var79[:, _var81]
_var82 = data_1[ordinal_integer_cols]
_var83 = [ordinal_float_data, _var82]
_var84 = 1
ordinal_data = pd.concat(_var83, axis=_var84)
_var85 = ordinal_data.columns
_var86 = 1
data_2 = data_1.drop(_var85, _var86)
_var87 = data_2.loc
_var88 = data_2.dtypes
_var89 = (_var88 == object)
nominal_string_data = _var87[:, _var89]
_var90 = data_2.loc
_var91 = data_2.dtypes
_var92 = (_var91 == int)
nominal_data = _var90[:, _var92]
data_3 = None
_var93 = nominal_string_data.columns
for i_0 in _var93:
    _var94 = preprocessing.LabelEncoder()
    _var95 = nominal_string_data[i_0]
    _var96 = _var94.fit_transform(_var95)
    nominal_string_data_0 = set_index_wrapper(nominal_string_data, i_0, _var96)
nominal_string_data_1 = __phi__(nominal_string_data_0, nominal_string_data)
_var97 = [nominal_data, nominal_string_data_1]
_var98 = 1
nominal_data_0 = pd.concat(_var97, axis=_var98)
_var99 = [ordinal_data, nominal_data_0]
_var100 = 1
data_4 = pd.concat(_var99, axis=_var100)
_var101 = ordinal_data.columns
_var102 = _var101.values
ordinalSelector = ItemSelector(_var102)
_var103 = nominal_data_0.columns
_var104 = _var103.values
nominalSelector = ItemSelector(_var104)
_var105 = 'union'
_var106 = 'ordinal'
_var107 = 'selector'
_var108 = (_var107, ordinalSelector)
_var109 = 'Imputer'
_var110 = (- 999)
_var111 = 'mean'
_var112 = preprocessing.Imputer(_var110, strategy=_var111)
_var113 = (_var109, _var112)
_var114 = [_var108, _var113]
_var115 = Pipeline(_var114)
_var116 = (_var106, _var115)
_var117 = 'nominal'
_var118 = 'selector'
_var119 = (_var118, nominalSelector)
_var120 = 'Imputer'
_var121 = (- 999)
_var122 = 'most_frequent'
_var123 = preprocessing.Imputer(_var121, strategy=_var122)
_var124 = (_var120, _var123)
_var125 = 'OneHot'
_var126 = False
_var127 = preprocessing.OneHotEncoder(sparse=_var126)
_var128 = (_var125, _var127)
_var129 = [_var119, _var124, _var128]
_var130 = Pipeline(_var129)
_var131 = (_var117, _var130)
_var132 = [_var116, _var131]
_var133 = FeatureUnion(transformer_list=_var132)
_var134 = (_var105, _var133)
_var135 = 'MinMaxScaler'
_var136 = (- 1)
_var137 = [_var136, 1]
_var138 = preprocessing.MinMaxScaler(_var137)
_var139 = (_var135, _var138)
_var140 = [_var134, _var139]
pipeline_preprocessing = Pipeline(_var140)
_var141 = 1
_var142 = 'binary'
recall_1_scorer = make_scorer(recall_score, pos_label=_var141, average=_var142)
_var143 = 0
_var144 = 'binary'
recall_0_scorer = make_scorer(recall_score, pos_label=_var143, average=_var144)
_var145 = 1
roc_curve_scorer = make_scorer(roc_curve, pos_label=_var145)
start = time.time()
_var146 = 'union'
_var147 = 'ordinal'
_var148 = 'selector'
_var149 = (_var148, ordinalSelector)
_var150 = 'Imputer'
_var151 = (- 999)
_var152 = 'mean'
_var153 = preprocessing.Imputer(_var151, strategy=_var152)
_var154 = (_var150, _var153)
_var155 = [_var149, _var154]
_var156 = Pipeline(_var155)
_var157 = (_var147, _var156)
_var158 = 'nominal'
_var159 = 'selector'
_var160 = (_var159, nominalSelector)
_var161 = 'Imputer'
_var162 = (- 999)
_var163 = 'most_frequent'
_var164 = preprocessing.Imputer(_var162, strategy=_var163)
_var165 = (_var161, _var164)
_var166 = 'OneHot'
_var167 = False
_var168 = preprocessing.OneHotEncoder(sparse=_var167)
_var169 = (_var166, _var168)
_var170 = [_var160, _var165, _var169]
_var171 = Pipeline(_var170)
_var172 = (_var158, _var171)
_var173 = [_var157, _var172]
_var174 = FeatureUnion(transformer_list=_var173)
_var175 = (_var146, _var174)
_var176 = 'MinMaxScaler'
_var177 = (- 1)
_var178 = [_var177, 1]
_var179 = preprocessing.MinMaxScaler(_var178)
_var180 = (_var176, _var179)
_var181 = 'GUS'
_var182 = GenericUnivariateSelect()
_var183 = (_var181, _var182)
_var184 = 'SGD'
_var185 = 'balanced'
_var186 = True
_var187 = SGDClassifier(class_weight=_var185, shuffle=_var186)
_var188 = (_var184, _var187)
_var189 = [_var175, _var180, _var183, _var188]
pipeline = Pipeline(_var189)
classes_names = ['Alumno no desertor(0)', 'Alumno desertor(1)']
_var194 = 'DESERTA_ALU'
_var195 = output_data[_var194]
_var196 = 0.5
(_var190, _var191, _var192, _var193) = train_test_split(data_4, _var195, test_size=_var196)
X_train = _var190
X_test = _var191
y_train = _var192
y_test = _var193
_var197 = 5
_var198 = 20
_var199 = sp_randint(_var197, _var198)
_var200 = 1
_var201 = 3
_var202 = sp_randint(_var200, _var201)
_var203 = ['l2']
_var204 = ['log', 'hinge', 'perceptron']
_var205 = ['k_best']
param_dist = {'GUS__param': _var199, 'SGD__n_iter': _var202, 'SGD__penalty': _var203, 'SGD__loss': _var204, 'GUS__mode': _var205}
param_distributions = param_dist
n_iter_search = 15
_var206 = 5
_var207 = True
cv_0 = StratifiedKFold(y_train, n_folds=_var206, shuffle=_var207)
_var208 = (- 1)
_var209 = 'roc_auc'
random_search = RandomizedSearchCV(pipeline, param_distributions=param_dist, n_iter=n_iter_search, n_jobs=_var208, scoring=_var209, cv=cv_0)
_var210 = 'Entrenando Modelo...'
print(_var210)
random_search_0 = random_search.fit(X_train, y_train)
_var211 = 'RandomizedSearchCV took %.2f seconds for %d candidates parameter settings.'
_var212 = time.time()
_var213 = (_var212 - start)
_var214 = (_var213, n_iter_search)
_var215 = (_var211 % _var214)
print(_var215)
_var216 = random_search_0.grid_scores_
report(_var216)
_var217 = '-------------------------------------------------'
print(_var217)
_var218 = 'Revisando Learning Error...'
print(_var218)
y_pred = random_search_0.predict(X_train)
_var219 = confusion_matrix(y_train, y_pred)
print(_var219)
_var220 = classification_report(y_train, y_pred, target_names=classes_names)
print(_var220)
_var221 = '-------------------------------------------------'
print(_var221)
_var222 = 'Realizando Prediccion...'
print(_var222)
y_pred_0 = random_search_0.predict(X_test)
_var223 = confusion_matrix(y_test, y_pred_0)
print(_var223)
_var224 = classification_report(y_test, y_pred_0, target_names=classes_names)
print(_var224)
_var225 = '-------------------------------------------------'
print(_var225)
_var230 = global_wrapper(y)
_var231 = 0.33
_var232 = None
_var233 = global_wrapper(y)
(_var226, _var227, _var228, _var229) = train_test_split(data_4, _var230, test_size=_var231, random_state=_var232, stratify=_var233)
X_train_outer = _var226
X_test_outer = _var227
y_train_outer = _var228
y_test_outer = _var229

def PerfGridSearchCV(estimator_0, param_distributions_0, X_train_0, y_train_0, X_test_0, y_test_0, n_iter_seach=5, scoring='roc_auc', n_folds=5, random_state=None, classes_names_0=['Alumno no desertor(0)', 'Alumno desertor(1)'], n_jobs_0=(- 1)):
    start_0 = time.time()
    _var234 = 'union'
    _var235 = 'ordinal'
    _var236 = 'selector'
    _var237 = global_wrapper(ordinalSelector)
    _var238 = (_var236, _var237)
    _var239 = 'Imputer'
    _var240 = (- 999)
    _var241 = 'mean'
    _var242 = preprocessing.Imputer(_var240, strategy=_var241)
    _var243 = (_var239, _var242)
    _var244 = [_var238, _var243]
    _var245 = Pipeline(_var244)
    _var246 = (_var235, _var245)
    _var247 = 'nominal'
    _var248 = 'selector'
    _var249 = global_wrapper(nominalSelector)
    _var250 = (_var248, _var249)
    _var251 = 'Imputer'
    _var252 = (- 999)
    _var253 = 'most_frequent'
    _var254 = preprocessing.Imputer(_var252, strategy=_var253)
    _var255 = (_var251, _var254)
    _var256 = 'OneHot'
    _var257 = False
    _var258 = 'ignore'
    _var259 = preprocessing.OneHotEncoder(sparse=_var257, handle_unknown=_var258)
    _var260 = (_var256, _var259)
    _var261 = [_var250, _var255, _var260]
    _var262 = Pipeline(_var261)
    _var263 = (_var247, _var262)
    _var264 = [_var246, _var263]
    _var265 = FeatureUnion(transformer_list=_var264)
    _var266 = (_var234, _var265)
    _var267 = 'MinMaxScaler'
    _var268 = (- 1)
    _var269 = [_var268, 1]
    _var270 = preprocessing.MinMaxScaler(_var269)
    _var271 = (_var267, _var270)
    _var272 = 'estimator'
    _var273 = (_var272, estimator_0)
    _var274 = [_var266, _var271, _var273]
    pipeline_0 = Pipeline(_var274)
    _var275 = True
    cv_1 = StratifiedKFold(y_train_0, n_folds=n_folds, shuffle=_var275, random_state=random_state)
    _var276 = global_wrapper(param_dist)
    _var277 = global_wrapper(n_iter_search)
    _var278 = 9
    random_search_1 = RandomizedSearchCV(pipeline_0, param_distributions=_var276, n_iter=_var277, n_jobs=n_jobs_0, scoring=scoring, cv=cv_1, verbose=_var278, random_state=random_state)
    _var279 = 'Training Model...'
    print(_var279)
    random_search_2 = random_search_1.fit(X_train_0, y_train_0)
    _var280 = 'RandomizedSearchCV took %.2f seconds for %d candidates parameter settings.'
    _var281 = time.time()
    _var282 = (_var281 - start_0)
    _var283 = global_wrapper(n_iter_search)
    _var284 = (_var282, _var283)
    _var285 = (_var280 % _var284)
    print(_var285)
    _var286 = random_search_2.grid_scores_
    report(_var286)
    _var287 = '-------------------------------------------------'
    print(_var287)
    _var288 = 'Report for Misclassification Training Error...'
    print(_var288)
    y_pred_1 = random_search_2.predict(X_train_0)
    _var289 = confusion_matrix(y_train_0, y_pred_1)
    print(_var289)
    _var290 = classification_report(y_train_0, y_pred_1, target_names=classes_names_0)
    print(_var290)
    _var291 = '-------------------------------------------------'
    print(_var291)
    _var292 = 'Report for Misclassification Testing Error...'
    print(_var292)
    y_pred_2 = random_search_2.predict(X_test_0)
    _var293 = confusion_matrix(y_test_0, y_pred_2)
    print(_var293)
    _var294 = classification_report(y_test_0, y_pred_2, target_names=classes_names_0)
    print(_var294)
    _var295 = '-------------------------------------------------'
    print(_var295)
    _var296 = random_search_2.best_estimator_
    return _var296
