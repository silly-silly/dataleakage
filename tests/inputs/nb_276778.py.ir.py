

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
from sklearn.svm import LinearSVC
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV, ParameterSampler
from sklearn.kernel_approximation import RBFSampler
from scipy.stats.distributions import expon
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.pipeline import Pipeline, FeatureUnion
import unbalanced_dataset as ud
from sklearn.cross_validation import train_test_split, cross_val_score, StratifiedKFold
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.calibration import CalibratedClassifierCV
import seaborn as sns
from operator import itemgetter
from scipy.stats import randint as sp_randint
sns.set()
import matplotlib
import matplotlib.pyplot as plt
from sklearn.learning_curve import learning_curve
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
_var79 = 1
_var80 = 'binary'
recall_1_scorer = make_scorer(recall_score, pos_label=_var79, average=_var80)
_var81 = 0
_var82 = 'binary'
recall_0_scorer = make_scorer(recall_score, pos_label=_var81, average=_var82)
_var83 = 1
roc_curve_scorer = make_scorer(roc_curve, pos_label=_var83)
classes_names = ['Alumno no desertor(0)', 'Alumno desertor(1)']
ordinal_integer_cols = ['CANT_TRASLADOS_ALU', 'CANT_DELITOS_COM_ALU', 'CANT_CURSOS_RBD', 'CANT_DELITOS_MANZANA_RBD', 'CANT_DOC_M_RBD', 'CANT_DOC_F_RBD', 'PAGO_MATRICULA_RBD', 'PAGO_MENSUAL_RBD', 'SEL_SNED_RBD', 'BECAS_DISP_RBD', 'PROM_ALU_CUR_RBD', 'CANT_DELITOS_COM_RBD', 'EDU_P', 'EDU_M', 'ING_HOGAR']
_var84 = data_1.loc
_var85 = data_1.dtypes
_var86 = (_var85 == float)
ordinal_float_data = _var84[:, _var86]
_var87 = data_1[ordinal_integer_cols]
_var88 = [ordinal_float_data, _var87]
_var89 = 1
ordinal_data = pd.concat(_var88, axis=_var89)
_var90 = ordinal_data.columns
_var91 = 1
data_2 = data_1.drop(_var90, _var91)
_var92 = data_2.loc
_var93 = data_2.dtypes
_var94 = (_var93 == object)
nominal_string_data = _var92[:, _var94]
_var95 = data_2.loc
_var96 = data_2.dtypes
_var97 = (_var96 == int)
nominal_data = _var95[:, _var97]
data_3 = None
_var98 = nominal_string_data.columns
for i_0 in _var98:
    _var99 = preprocessing.LabelEncoder()
    _var100 = nominal_string_data[i_0]
    _var101 = _var99.fit_transform(_var100)
    nominal_string_data_0 = set_index_wrapper(nominal_string_data, i_0, _var101)
nominal_string_data_1 = __phi__(nominal_string_data_0, nominal_string_data)
_var102 = [nominal_data, nominal_string_data_1]
_var103 = 1
nominal_data_0 = pd.concat(_var102, axis=_var103)
_var104 = [ordinal_data, nominal_data_0]
_var105 = 1
data_4 = pd.concat(_var104, axis=_var105)
_var106 = ordinal_data.columns
_var107 = _var106.values
ordinalSelector = ItemSelector(_var107)
_var108 = nominal_data_0.columns
_var109 = _var108.values
nominalSelector = ItemSelector(_var109)
_var110 = 'union'
_var111 = 'ordinal'
_var112 = 'selector'
_var113 = (_var112, ordinalSelector)
_var114 = 'Imputer'
_var115 = (- 999)
_var116 = 'mean'
_var117 = preprocessing.Imputer(_var115, strategy=_var116)
_var118 = (_var114, _var117)
_var119 = [_var113, _var118]
_var120 = Pipeline(_var119)
_var121 = (_var111, _var120)
_var122 = 'nominal'
_var123 = 'selector'
_var124 = (_var123, nominalSelector)
_var125 = 'Imputer'
_var126 = (- 999)
_var127 = 'most_frequent'
_var128 = preprocessing.Imputer(_var126, strategy=_var127)
_var129 = (_var125, _var128)
_var130 = [_var124, _var129]
_var131 = Pipeline(_var130)
_var132 = (_var122, _var131)
_var133 = [_var121, _var132]
_var134 = FeatureUnion(transformer_list=_var133)
_var135 = (_var110, _var134)
_var136 = [_var135]
pipeline_preprocessing = Pipeline(_var136)
_var137 = 'balanced'
_var138 = (- 1)
_var139 = 60
clf = ExtraTreesClassifier(class_weight=_var137, n_jobs=_var138, n_estimators=_var139)
data_pre = pipeline_preprocessing.fit_transform(data_4)
X_1 = data_pre
_var140 = 'DESERTA_ALU'
y_1 = output_data[_var140]
_var145 = 0.5
(_var141, _var142, _var143, _var144) = train_test_split(X_1, y_1, test_size=_var145)
X_train = _var141
X_test = _var142
y_train = _var143
y_test = _var144
_var146 = 1
_var147 = 11
_var148 = sp_randint(_var146, _var147)
_var149 = 1
_var150 = 11
_var151 = sp_randint(_var149, _var150)
_var152 = 1
_var153 = 11
_var154 = sp_randint(_var152, _var153)
_var155 = [True, False]
_var156 = ['gini', 'entropy']
param_dist = {'max_depth': _var148, 'min_samples_split': _var151, 'min_samples_leaf': _var154, 'bootstrap': _var155, 'criterion': _var156}
n_iter_search = 20
_var157 = 2
_var158 = 3
random_search = RandomizedSearchCV(clf, param_distributions=param_dist, n_iter=n_iter_search, n_jobs=_var157, scoring=recall_1_scorer, cv=_var158)
start = time.time()
_var159 = 'Entrenando Modelo...'
print(_var159)
random_search_0 = random_search.fit(X_train, y_train)
_var160 = 'RandomizedSearchCV took %.2f seconds for %d candidates parameter settings.'
_var161 = time.time()
_var162 = (_var161 - start)
_var163 = (_var162, n_iter_search)
_var164 = (_var160 % _var163)
print(_var164)
_var165 = random_search_0.grid_scores_
report(_var165)
_var166 = 'Realizando Prediccion...'
print(_var166)
y_pred = random_search_0.predict(X_test)
_var167 = confusion_matrix(y_test, y_pred)
print(_var167)
_var168 = classification_report(y_test, y_pred, target_names=classes_names)
print(_var168)
_var169 = '--- %s seconds ---'
_var170 = time.time()
_var171 = (_var170 - start)
_var172 = (_var169 % _var171)
print(_var172)
cantidad = 30
_var173 = random_search_0.best_estimator_
importances = _var173.feature_importances_
_var174 = [tree.feature_importances_ for tree in random_search.best_estimator_.estimators_]
_var175 = 0
std = np.std(_var174, axis=_var175)
_var176 = np.argsort(importances)
_var177 = (- 1)
indices = _var176[::_var177]
_var178 = 0
indices_0 = indices[_var178:cantidad]
plt.figure()
_var179 = 'Importancia de los Atributos'
plt.title(_var179)
_var180 = importances[indices_0]
_var181 = _var180.shape
_var182 = 0
_var183 = _var181[_var182]
_var184 = range(_var183)
_var185 = list(_var184)
_var186 = importances[indices_0]
_var187 = 'gray'
_var188 = 'center'
plt.barh(_var185, _var186, color=_var187, align=_var188)
_var189 = importances[indices_0]
_var190 = _var189.shape
_var191 = 0
_var192 = _var190[_var191]
_var193 = range(_var192)
_var194 = list(_var193)
_var195 = data_4.columns
_var196 = _var195.values
_var197 = 'horizontal'
plt.yticks(_var194, _var196, rotation=_var197)
_var198 = (- 1)
_var199 = importances[indices_0]
_var200 = _var199.shape
_var201 = 0
_var202 = _var200[_var201]
_var203 = [_var198, _var202]
plt.ylim(_var203)
_var204 = 'Atributo'
plt.ylabel(_var204)
_var205 = 'Importancia'
plt.xlabel(_var205)
plt.show()
_var206 = random_search_0.best_estimator_
_var207 = 'prefit'
c = CalibratedClassifierCV(_var206, cv=_var207)
