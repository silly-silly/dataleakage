

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
import pandas as pd
import numpy as np
_var0 = 'display.max_columns'
_var1 = None
pd.set_option(_var0, _var1)
import sklearn.preprocessing as preprocessing
import sklearn.feature_extraction as feature_extraction
from sklearn_pandas import DataFrameMapper, cross_val_score
from sklearn import cross_validation
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.metrics import classification_report
_var2 = '/Users/Felipe/PycharmProjects/FracasoEscolarChile/DatasetsProcesados/SIMCE/ALU/SIMCE_GEO_2013-2014.csv'
_var3 = 0
_var4 = '|'
_var5 = '.'
r = pd.read_csv(_var2, header=_var3, sep=_var4, decimal=_var5)
cols = ['MRUN', 'COD_COM_ALU', 'NOM_COM_ALU', 'SIT_FIN_R', 'EDAD_ALU', 'CODINE11', 'LAT_MANZANA_ALU', 'LON_MANZANA_ALU', 'RIESGO_DESERCION_RBD', 'DIR_RBD', 'LAT_MANZANA_RBD', 'LON_MANZANA_RBD', 'CONVIVENCIA_2M_RBD', 'CONVIVENCIA_4B_RBD', 'CONVIVENCIA_6B_RBD', 'AUTOESTIMA_MOTIVACION_2M_RBD', 'AUTOESTIMA_MOTIVACION_4B_RBD', 'AUTOESTIMA_MOTIVACION_6B_RBD', 'AUTOESTIMA_MOTIVACION_8B_RBD', 'PARTICIPACION_2M_RBD', 'PARTICIPACION_4B_RBD', 'PARTICIPACION_6B_RBD', 'PARTICIPACION_8B_RBD', 'IVE_MEDIA_RBD', 'IVE_BASICA_RBD', 'PSU_PROM_2013_RBD', 'CPAD_DISP', 'DGV_RBD', 'NOM_RBD', 'NOM_COM_RBD', 'COD_COM_RBD', 'RBD', 'PROM_GRAL', 'ASISTENCIA', 'LET_CUR', 'CLASIFICACION_SEP_RBD']
_var6 = 1
data = r.drop(cols, _var6)
_var7 = data.columns
_var8 = [0, 21]
_var9 = _var7[_var8]
_var10 = 1
data_0 = data.drop(_var9, _var10)
nocat = ['CANT_TRASLADOS_ALU', 'CANT_DELITOS_COM_ALU', 'CANT_DELITOS_MANZANA_RBD', 'CANT_DOC_M_RBD', 'CANT_DOC_F_RBD', 'CANT_DOC_RBD', 'POB_FLOT_RBD', 'BECAS_DISP_RBD', 'MAT_TOTAL_RBD', 'VACANTES_CUR_IN_RBD', 'PROM_ALU_CUR_RBD', 'CANT_DELITOS_COM_RBD', 'EDU_P', 'EDU_M', 'ING_HOGAR', 'CANT_CURSOS_RBD', 'PAGO_MATRICULA_RBD', 'PAGO_MENSUAL_RBD']
_var11 = ['ABANDONA_ALU', 'DESERTA_ALU', 'ABANDONA_2014_ALU']
output_data = data_0[_var11]
_var12 = ['ABANDONA_ALU', 'DESERTA_ALU', 'ABANDONA_2014_ALU']
_var13 = 1
data_1 = data_0.drop(_var12, _var13)
_var14 = data_1.loc
_var15 = data_1.dtypes
_var16 = (_var15 == float)
float_data = _var14[:, _var16]
_var17 = data_1.loc
_var18 = data_1.dtypes
_var19 = (_var18 == object)
object_data = _var17[:, _var19]
_var20 = data_1.loc
_var21 = data_1.dtypes
_var22 = (_var21 == int)
_var23 = _var20[:, _var22]
_var24 = 1
categorical_data = _var23.drop(nocat, _var24)
_var25 = ['GEN_ALU', 'ORI_RELIGIOSA_RBD']
categorical_data_0 = categorical_data[_var25]
_var26 = data_1.loc
_var27 = data_1.dtypes
_var28 = (_var27 == int)
_var29 = _var26[:, _var28]
integer_data = _var29[nocat]
_var30 = float_data.columns
_var31 = _var30.values
_var32 = np.nan
_var33 = preprocessing.Imputer(missing_values=_var32)
_var34 = 0
_var35 = 1
_var36 = (_var34, _var35)
_var37 = preprocessing.MinMaxScaler(_var36)
_var38 = [_var33, _var37]
_var39 = (_var31, _var38)
_var40 = [_var39]
float_data_mapper = DataFrameMapper(_var40)
_var41 = integer_data.columns
_var42 = _var41.values
_var43 = np.nan
_var44 = preprocessing.Imputer(missing_values=_var43)
_var45 = 0
_var46 = 1
_var47 = (_var45, _var46)
_var48 = preprocessing.MinMaxScaler(_var47)
_var49 = [_var44, _var48]
_var50 = (_var42, _var49)
_var51 = [_var50]
integer_data_mapper = DataFrameMapper(_var51)
_var52 = categorical_data_0.columns
_var53 = _var52.values
_var54 = (- 999)
_var55 = preprocessing.Imputer(missing_values=_var54)
_var56 = (_var53, _var55)
_var57 = [_var56]
categorical_data_mapper = DataFrameMapper(_var57)
_var58 = 'Imputando : 25%...'
print(_var58)
float_data_arr = float_data_mapper.fit_transform(float_data)
_var59 = float_data.columns
float_data_0 = pd.DataFrame(float_data_arr, columns=_var59)
_var60 = 'Imputando : 50%...'
print(_var60)
integer_data_arr = integer_data_mapper.fit_transform(integer_data)
_var61 = integer_data.columns
integer_data_0 = pd.DataFrame(integer_data_arr, columns=_var61)
_var62 = 'Imputando : 100%...'
print(_var62)
_var63 = 'Vectorizando : 0%...'
print(_var63)
categorical_data_arr = categorical_data_mapper.fit_transform(categorical_data_0)
_var64 = categorical_data_0.columns
categorical_data_1 = pd.DataFrame(categorical_data_arr, columns=_var64)
_var65 = False
object_data_vectorizer = feature_extraction.DictVectorizer(sparse=_var65)
_var66 = object_data.T
_var67 = _var66.to_dict()
_var68 = _var67.values()
_var69 = list(_var68)
object_data_prep = object_data_vectorizer.fit_transform(_var69)
_var70 = object_data_vectorizer.get_feature_names()
object_data_prep_df = pd.DataFrame(object_data_prep, columns=_var70)
_var71 = 'Vectorizando : 50%...'
print(_var71)
_var72 = False
categorical_data_vectorizer = feature_extraction.DictVectorizer(sparse=_var72)
_var73 = categorical_data_1.applymap(str)
_var74 = _var73.T
_var75 = _var74.to_dict()
_var76 = _var75.values()
_var77 = list(_var76)
categorical_data_prep = categorical_data_vectorizer.fit_transform(_var77)
_var78 = categorical_data_vectorizer.get_feature_names()
categorical_data_prep_df = pd.DataFrame(categorical_data_prep, columns=_var78)
_var79 = 'Vectorizando : 100%'
print(_var79)
_var80 = [float_data_0, integer_data_0, object_data_prep_df]
_var81 = 1
_var82 = 'inner'
input_data = pd.concat(_var80, axis=_var81, join=_var82)
X = np.array(input_data)
_var83 = 'DESERTA_ALU'
_var84 = output_data[_var83]
y = np.array(_var84)
t_names = ['Alumno Retenido(0)', 'Alumno Desertor(1)']
_var89 = 0.7
_var90 = 0
(_var85, _var86, _var87, _var88) = cross_validation.train_test_split(X, y, test_size=_var89, random_state=_var90)
X_train = _var85
X_test = _var86
y_train = _var87
y_test = _var88
_var91 = 'log'
_var92 = 'l2'
_var93 = (- 1)
_var94 = 'auto'
estimator_log = SGDClassifier(loss=_var91, penalty=_var92, n_jobs=_var93, class_weight=_var94)
_var95 = 'hinge'
_var96 = 'l2'
_var97 = (- 1)
_var98 = 'auto'
estimator_linear = SGDClassifier(loss=_var95, penalty=_var96, n_jobs=_var97, class_weight=_var98)
_var99 = 'perceptron'
_var100 = 'l2'
_var101 = (- 1)
_var102 = 'auto'
estimator_per = SGDClassifier(loss=_var99, penalty=_var100, n_jobs=_var101, class_weight=_var102)
_var103 = 'modified_huber'
_var104 = 'l2'
_var105 = (- 1)
_var106 = 'auto'
estimator_mh = SGDClassifier(loss=_var103, penalty=_var104, n_jobs=_var105, class_weight=_var106)
_var107 = 'squared_hinge'
_var108 = 'l2'
_var109 = (- 1)
_var110 = 'auto'
estimator_h = SGDClassifier(loss=_var107, penalty=_var108, n_jobs=_var109, class_weight=_var110)
_var111 = 'huber'
_var112 = 'l2'
_var113 = (- 1)
_var114 = 'auto'
estimator_hu = SGDClassifier(loss=_var111, penalty=_var112, n_jobs=_var113, class_weight=_var114)
_var115 = [estimator_log, estimator_linear, estimator_h, estimator_hu, estimator_mh, estimator_per]
for estimator in _var115:
    estimator_0 = estimator.fit(X_train, y_train)
    y_pred = estimator_0.predict(X_test)
    m = confusion_matrix(y_test, y_pred)
    _var116 = 'Modelo : '
    _var117 = estimator_0.loss
    _var118 = (_var116 + _var117)
    print(_var118)
    _var119 = classification_report(y_test, y_pred, target_names=t_names)
    print(_var119)
    _var120 = '\n'
    print(_var120)
estimator_1 = __phi__(estimator_0, estimator)
gnb = GaussianNB()
gnb_0 = gnb.fit(X_train, y_train)
y_pred_0 = gnb_0.predict(X_test)
m_0 = confusion_matrix(y_test, y_pred_0)
_var121 = 'Modelo : Naive-Bayes'
print(_var121)
_var122 = classification_report(y_test, y_pred_0, target_names=t_names)
print(_var122)
_var123 = '\n'
print(_var123)
