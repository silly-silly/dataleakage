

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
import matplotlib
import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn import svm
from sklearn import cross_validation
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import RFE
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from random import randrange, choice
from sklearn.neighbors import NearestNeighbors
from unbalanced_dataset import SMOTE
import seaborn as sns
_var3 = 'display.max_columns'
_var4 = None
pd.set_option(_var3, _var4)
_var5 = '/Users/Felipe/PycharmProjects/FracasoEscolarChile/DatasetsProcesados/SIMCE/ALU/SIMCE_GEO_2013-2014.csv'
_var6 = '|'
_var7 = '.'
r = pd.read_csv(_var5, sep=_var6, decimal=_var7)
_var8 = '/Users/Felipe/PycharmProjects/FracasoEscolarChile/DatasetsProcesados/RBD_GEO_2013_MUESTRA.csv'
_var9 = ','
_var10 = '.'
r2 = pd.read_csv(_var8, sep=_var9, decimal=_var10)
_var11 = r.columns
_var12 = 0
_var13 = 3
_var14 = _var11[_var12:_var13]
_var15 = 1
r_0 = r.drop(_var14, _var15)
_var16 = r_0.GSE_MANZANA_ALU
_var17 = 0
_var18 = (_var16 > _var17)
_var19 = r_0.CPAD_DISP
_var20 = 1
_var21 = (_var19 == _var20)
_var22 = (_var18 & _var21)
r_1 = r_0[_var22]
_var23 = 'inner'
_var24 = ['RBD']
c = pd.merge(r_1, r2, how=_var23, on=_var24)
_var25 = preprocessing.LabelEncoder()
_var26 = r_1.SIT_FIN_R
_var27 = _var25.fit_transform(_var26)
r_2 = set_field_wrapper(r_1, 'SIT_FIN_R', _var27)
_var28 = preprocessing.LabelEncoder()
_var29 = r_2.LET_CUR
_var30 = _var28.fit_transform(_var29)
r_3 = set_field_wrapper(r_2, 'LET_CUR', _var30)
_var31 = preprocessing.LabelEncoder()
_var32 = r_3.GSE_MANZANA_ALU
_var33 = _var31.fit_transform(_var32)
r_4 = set_field_wrapper(r_3, 'GSE_MANZANA_ALU', _var33)
_var34 = 'IVE_POND'
_var35 = ['IVE_BASICA_RBD', 'IVE_MEDIA_RBD']
_var36 = c[_var35]

def _func0(x):
    _var37 = x.mean()
    return _var37
_var38 = 1
_var39 = _var36.apply(_func0, axis=_var38)
c_0 = set_index_wrapper(c, _var34, _var39)
_var40 = 'CONVIVENCIA_POND'
_var41 = 'CONVIVENCIA'
_var42 = c_0.filter(regex=_var41)

def _func1(x_0):
    _var43 = x_0.mean()
    return _var43
_var44 = 1
_var45 = _var42.apply(_func1, axis=_var44)
c_1 = set_index_wrapper(c_0, _var40, _var45)
_var46 = 'AUTOESTIMA_MOTIVACION_POND'
_var47 = 'AUTOESTIMA_MOTIVACION'
_var48 = c_1.filter(regex=_var47)

def _func2(x_1):
    _var49 = x_1.mean()
    return _var49
_var50 = 1
_var51 = _var48.apply(_func2, axis=_var50)
c_2 = set_index_wrapper(c_1, _var46, _var51)
_var52 = 'PARTICIPACION_POND'
_var53 = 'PARTICIPACION'
_var54 = c_2.filter(regex=_var53)

def _func3(x_2):
    _var55 = x_2.mean()
    return _var55
_var56 = 1
_var57 = _var54.apply(_func3, axis=_var56)
c_3 = set_index_wrapper(c_2, _var52, _var57)
_var58 = ['COD_ENSE', 'COD_GRADO', 'COD_JOR', 'GEN_ALU', 'COD_COM_ALU', 'REPITENTE_ALU', 'ABANDONA_ALU', 'SOBRE_EDAD_ALU', 'CANT_TRASLADOS_ALU', 'DIST_ALU_A_RBD_C', 'DIST_ALU_A_RBD', 'DESERTA_ALU', 'EDU_M', 'EDU_P', 'ING_HOGAR', 'TASA_ABANDONO_RBD', 'TASA_REPITENCIA_RBD', 'TASA_TRASLADOS_RBD', 'IAV_MANZANA_RBD', 'CULT_MANZANA_RBD', 'DISP_GSE_MANZANA_RBD', 'DEL_DROG_MANZANA_RBD', 'CANT_CURSOS_RBD', 'CANT_DELITOS_MANZANA_RBD', 'PROF_AULA_H_MAT_RBD', 'PROF_TAXI_H_MAT_RBD', 'CONVIVENCIA_POND', 'AUTOESTIMA_MOTIVACION_POND', 'PARTICIPACION_POND', 'PORC_HORAS_LECTIVAS_DOC_RBD', 'PROM_EDAD_TITULACION_DOC_RBD', 'PROM_EDAD_DOC_RBD', 'PROM_ANOS_SERVICIO_DOC_RBD', 'PROM_ANOS_ESTUDIOS_DOC_RBD', 'CANT_DOC_RBD', 'PAGO_MATRICULA_RBD', 'PAGO_MENSUAL_RBD', 'IVE_POND']
r_5 = c_3[_var58]
r_6 = r_5.dropna()

def f(x_3):
    _var59 = x_3.name
    _var60 = 'ABANDONA_ALU'
    _var61 = (_var59 != _var60)
    if _var61:
        _var62 = (- 1)
        _var63 = 1
        _var64 = (_var62, _var63)
        min_max_scaler = preprocessing.MinMaxScaler(_var64)
        _var65 = min_max_scaler.fit_transform(x_3)
        return _var65
    return x_3
_var66 = 0
r_7 = r_6.apply(f, axis=_var66)
_var67 = ['DESERTA_ALU', 'ABANDONA_ALU']
_var68 = 1
_var69 = r_7.drop(_var67, _var68)
X = np.array(_var69)
_var70 = 'ABANDONA_ALU'
_var71 = r_7[_var70]
y = np.array(_var71)
_var76 = 0.33
(_var72, _var73, _var74, _var75) = cross_validation.train_test_split(X, y, test_size=_var76)
X_train = _var72
X_test = _var73
y_train = _var74
y_test = _var75
_var77 = 'log'
_var78 = 'l2'
_var79 = (- 1)
_var80 = 'auto'
estimator_log = SGDClassifier(loss=_var77, penalty=_var78, n_jobs=_var79, class_weight=_var80)
_var81 = 'hinge'
_var82 = 'l2'
_var83 = (- 1)
_var84 = 'auto'
estimator_linear = SGDClassifier(loss=_var81, penalty=_var82, n_jobs=_var83, class_weight=_var84)
_var85 = 'perceptron'
_var86 = 'l2'
_var87 = (- 1)
_var88 = 'auto'
estimator_per = SGDClassifier(loss=_var85, penalty=_var86, n_jobs=_var87, class_weight=_var88)
_var89 = 'modified_huber'
_var90 = 'l2'
_var91 = (- 1)
_var92 = 'auto'
estimator_mh = SGDClassifier(loss=_var89, penalty=_var90, n_jobs=_var91, class_weight=_var92)
_var93 = 'squared_hinge'
_var94 = 'l2'
_var95 = (- 1)
_var96 = 'auto'
estimator_h = SGDClassifier(loss=_var93, penalty=_var94, n_jobs=_var95, class_weight=_var96)
_var97 = 'huber'
_var98 = 'l2'
_var99 = (- 1)
_var100 = 'auto'
estimator_hu = SGDClassifier(loss=_var97, penalty=_var98, n_jobs=_var99, class_weight=_var100)
_var101 = [estimator_log, estimator_linear, estimator_h, estimator_hu, estimator_mh, estimator_per]
for estimator in _var101:
    estimator_0 = estimator.fit(X_train, y_train)
    y_pred = estimator_0.predict(X_test)
    m = confusion_matrix(y_test, y_pred)
    _var102 = '\n'
    print(_var102)
    _var103 = 'Modelo : '
    _var104 = estimator_0.loss
    _var105 = (_var103 + _var104)
    print(_var105)
    _var106 = 'Matriz de Confusion : '
    print(_var106)
    print(m)
    _var107 = 'Precision Total de %f, un %f en la retencion(Clase 0) y %f en la desercion(Clase 1).'
    _var108 = 0
    _var109 = m[_var108]
    _var110 = 0
    _var111 = _var109[_var110]
    _var112 = 1
    _var113 = m[_var112]
    _var114 = 1
    _var115 = _var113[_var114]
    _var116 = (_var111 + _var115)
    _var117 = 0
    _var118 = m[_var117]
    _var119 = 0
    _var120 = _var118[_var119]
    _var121 = 0
    _var122 = m[_var121]
    _var123 = 1
    _var124 = _var122[_var123]
    _var125 = (_var120 + _var124)
    _var126 = 1
    _var127 = m[_var126]
    _var128 = 1
    _var129 = _var127[_var128]
    _var130 = (_var125 + _var129)
    _var131 = 1
    _var132 = m[_var131]
    _var133 = 0
    _var134 = _var132[_var133]
    _var135 = (_var130 + _var134)
    _var136 = (_var116 / _var135)
    _var137 = 0
    _var138 = m[_var137]
    _var139 = 0
    _var140 = _var138[_var139]
    _var141 = 0
    _var142 = m[_var141]
    _var143 = 0
    _var144 = _var142[_var143]
    _var145 = 0
    _var146 = m[_var145]
    _var147 = 1
    _var148 = _var146[_var147]
    _var149 = (_var144 + _var148)
    _var150 = (_var140 / _var149)
    _var151 = 1
    _var152 = m[_var151]
    _var153 = 1
    _var154 = _var152[_var153]
    _var155 = 1
    _var156 = m[_var155]
    _var157 = 1
    _var158 = _var156[_var157]
    _var159 = 1
    _var160 = m[_var159]
    _var161 = 0
    _var162 = _var160[_var161]
    _var163 = (_var158 + _var162)
    _var164 = (_var154 / _var163)
    _var165 = (_var136, _var150, _var164)
    _var166 = (_var107 % _var165)
    _var167 = 1
    _var168 = (_var166 * _var167)
    print(_var168)
estimator_1 = __phi__(estimator_0, estimator)
gnb = GaussianNB()
gnb_0 = gnb.fit(X_train, y_train)
y_pred_0 = gnb_0.predict(X_test)
m_0 = confusion_matrix(y_test, y_pred_0)
_var169 = '\n'
print(_var169)
_var170 = 'Modelo : Naive-Bayes'
print(_var170)
_var171 = 'Matriz de Confusion : '
print(_var171)
print(m_0)
_var172 = 'Precision Total de %f, un %f en la retencion(Clase 0) y %f en la desercion(Clase 1).'
_var173 = 0
_var174 = m_0[_var173]
_var175 = 0
_var176 = _var174[_var175]
_var177 = 1
_var178 = m_0[_var177]
_var179 = 1
_var180 = _var178[_var179]
_var181 = (_var176 + _var180)
_var182 = 0
_var183 = m_0[_var182]
_var184 = 0
_var185 = _var183[_var184]
_var186 = 0
_var187 = m_0[_var186]
_var188 = 1
_var189 = _var187[_var188]
_var190 = (_var185 + _var189)
_var191 = 1
_var192 = m_0[_var191]
_var193 = 1
_var194 = _var192[_var193]
_var195 = (_var190 + _var194)
_var196 = 1
_var197 = m_0[_var196]
_var198 = 0
_var199 = _var197[_var198]
_var200 = (_var195 + _var199)
_var201 = (_var181 / _var200)
_var202 = 0
_var203 = m_0[_var202]
_var204 = 0
_var205 = _var203[_var204]
_var206 = 0
_var207 = m_0[_var206]
_var208 = 0
_var209 = _var207[_var208]
_var210 = 0
_var211 = m_0[_var210]
_var212 = 1
_var213 = _var211[_var212]
_var214 = (_var209 + _var213)
_var215 = (_var205 / _var214)
_var216 = 1
_var217 = m_0[_var216]
_var218 = 1
_var219 = _var217[_var218]
_var220 = 1
_var221 = m_0[_var220]
_var222 = 1
_var223 = _var221[_var222]
_var224 = 1
_var225 = m_0[_var224]
_var226 = 0
_var227 = _var225[_var226]
_var228 = (_var223 + _var227)
_var229 = (_var219 / _var228)
_var230 = (_var201, _var215, _var229)
_var231 = (_var172 % _var230)
_var232 = 1
_var233 = (_var231 * _var232)
print(_var233)
X.shape
y.shape
