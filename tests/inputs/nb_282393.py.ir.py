

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
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import random
from sklearn import preprocessing
import gc
from scipy.stats import skew, boxcox
from scipy import sparse
from sklearn.metrics import log_loss
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
_var0 = get_ipython()
_var1 = 'matplotlib'
_var2 = 'inline'
_var0.run_line_magic(_var1, _var2)
seed = 2017
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU, LeakyReLU, ELU, ParametricSoftplus, ThresholdedReLU, SReLU
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
from keras.optimizers import SGD, Nadam
from keras.regularizers import WeightRegularizer, ActivityRegularizer, l2, activity_l2
from keras.utils.np_utils import to_categorical
names = ['low_0', 'medium_0', 'high_0', 'low_1', 'medium_1', 'high_1', 'low_2', 'medium_2', 'high_2', 'low_3', 'medium_3', 'high_3', 'low_4', 'medium_4', 'high_4', 'low_5', 'medium_5', 'high_5', 'low_6', 'medium_6', 'high_6', 'low_7', 'medium_7', 'high_7', 'low_8', 'medium_8', 'high_8', 'low_9', 'medium_9', 'high_9']
data_path = '../2nd/'
total_col = 0
_var3 = 'train_blend_RFC_gini_BM_MB_add03052240_2017-03-10-22-02'
_var4 = '.csv'
file_train = (_var3 + _var4)
_var5 = 'test_blend_RFC_gini_mean_BM_MB_add03052240_2017-03-10-22-02'
_var6 = '.csv'
file_test_mean = (_var5 + _var6)
_var7 = 'test_blend_RFC_gini_gmean_BM_MB_add03052240_2017-03-10-22-02'
_var8 = '.csv'
file_test_gmean = (_var7 + _var8)
_var9 = (data_path + file_train)
_var10 = None
train_rfc_gini = pd.read_csv(_var9, header=_var10)
_var11 = (data_path + file_test_mean)
_var12 = None
test_rfc_gini_mean = pd.read_csv(_var11, header=_var12)
_var13 = (data_path + file_test_gmean)
_var14 = None
test_rfc_gini_gmean = pd.read_csv(_var13, header=_var14)
_var15 = train_rfc_gini.shape
_var16 = 1
n_column = _var15[_var16]
total_col_0 = (total_col + n_column)
_var17 = [('rfc_gini_' + x) for x in names[:n_column]]
train_rfc_gini_0 = set_field_wrapper(train_rfc_gini, 'columns', _var17)
_var18 = [('rfc_gini_' + x) for x in names[:n_column]]
test_rfc_gini_mean_0 = set_field_wrapper(test_rfc_gini_mean, 'columns', _var18)
_var19 = [('rfc_gini_' + x) for x in names[:n_column]]
test_rfc_gini_gmean_0 = set_field_wrapper(test_rfc_gini_gmean, 'columns', _var19)
_var20 = 'train_blend_RFC_entropy_BM_MB_add03052240_2017-03-10-21-10'
_var21 = '.csv'
file_train_0 = (_var20 + _var21)
_var22 = 'test_blend_RFC_entropy_mean_BM_MB_add03052240_2017-03-10-21-10'
_var23 = '.csv'
file_test_mean_0 = (_var22 + _var23)
_var24 = 'test_blend_RFC_entropy_gmean_BM_MB_add03052240_2017-03-10-21-10'
_var25 = '.csv'
file_test_gmean_0 = (_var24 + _var25)
_var26 = (data_path + file_train_0)
_var27 = None
train_rfc_entropy = pd.read_csv(_var26, header=_var27)
_var28 = (data_path + file_test_mean_0)
_var29 = None
test_rfc_entropy_mean = pd.read_csv(_var28, header=_var29)
_var30 = (data_path + file_test_gmean_0)
_var31 = None
test_rfc_entropy_gmean = pd.read_csv(_var30, header=_var31)
_var32 = train_rfc_entropy.shape
_var33 = 1
n_column_0 = _var32[_var33]
total_col_1 = (total_col_0 + n_column_0)
_var34 = [('rfc_entropy_' + x) for x in names[:n_column]]
train_rfc_entropy_0 = set_field_wrapper(train_rfc_entropy, 'columns', _var34)
_var35 = [('rfc_entropy_' + x) for x in names[:n_column]]
test_rfc_entropy_mean_0 = set_field_wrapper(test_rfc_entropy_mean, 'columns', _var35)
_var36 = [('rfc_entropy_' + x) for x in names[:n_column]]
test_rfc_entropy_gmean_0 = set_field_wrapper(test_rfc_entropy_gmean, 'columns', _var36)
_var37 = 'train_rfc_gini: {}\t test_rfc_gini_mean:{}\t test_rfc_gini_gmean:{}'
_var38 = train_rfc_gini_0.shape
_var39 = test_rfc_gini_mean_0.shape
_var40 = test_rfc_gini_gmean_0.shape
_var41 = _var37.format(_var38, _var39, _var40)
print(_var41)
_var42 = '\ntrain_rfc_entropy: {}\t test_rfc_entropy_mean:{}\t test_rfc_entropy_gmean:{}'
_var43 = train_rfc_entropy_0.shape
_var44 = test_rfc_entropy_mean_0.shape
_var45 = test_rfc_entropy_gmean_0.shape
_var46 = _var42.format(_var43, _var44, _var45)
print(_var46)
_var47 = '\ntrain_rfc_gini'
print(_var47)
_var48 = train_rfc_gini_0.iloc
_var49 = 5
_var50 = 3
_var51 = _var48[:_var49, :_var50]
print(_var51)
_var52 = '\ntrain_rfc_entropy'
print(_var52)
_var53 = train_rfc_entropy_0.iloc
_var54 = 5
_var55 = 3
_var56 = _var53[:_var54, :_var55]
print(_var56)
_var57 = 'train_blend_RFC_gini_BM_0322_2017-03-22-17-12'
_var58 = '.csv'
file_train_1 = (_var57 + _var58)
_var59 = 'test_blend_RFC_gini_mean_BM_0322_2017-03-22-17-12'
_var60 = '.csv'
file_test_mean_1 = (_var59 + _var60)
_var61 = 'test_blend_RFC_gini_gmean_BM_0322_2017-03-22-17-12'
_var62 = '.csv'
file_test_gmean_1 = (_var61 + _var62)
_var63 = (data_path + file_train_1)
_var64 = None
train_rfc_gini_0322 = pd.read_csv(_var63, header=_var64)
_var65 = (data_path + file_test_mean_1)
_var66 = None
test_rfc_gini_mean_0322 = pd.read_csv(_var65, header=_var66)
_var67 = (data_path + file_test_gmean_1)
_var68 = None
test_rfc_gini_gmean_0322 = pd.read_csv(_var67, header=_var68)
_var69 = train_rfc_gini_0322.shape
_var70 = 1
n_column_1 = _var69[_var70]
total_col_2 = (total_col_1 + n_column_1)
_var71 = [('rfc_gini_0322_' + x) for x in names[:n_column]]
train_rfc_gini_0322_0 = set_field_wrapper(train_rfc_gini_0322, 'columns', _var71)
_var72 = [('rfc_gini_0322_' + x) for x in names[:n_column]]
test_rfc_gini_mean_0322_0 = set_field_wrapper(test_rfc_gini_mean_0322, 'columns', _var72)
_var73 = [('rfc_gini_0322_' + x) for x in names[:n_column]]
test_rfc_gini_gmean_0322_0 = set_field_wrapper(test_rfc_gini_gmean_0322, 'columns', _var73)
_var74 = 'train_blend_RFC_entropy_BM_0322_2017-03-22-16-02'
_var75 = '.csv'
file_train_2 = (_var74 + _var75)
_var76 = 'test_blend_RFC_entropy_mean_BM_0322_2017-03-22-16-02'
_var77 = '.csv'
file_test_mean_2 = (_var76 + _var77)
_var78 = 'test_blend_RFC_entropy_gmean_BM_0322_2017-03-22-16-02'
_var79 = '.csv'
file_test_gmean_2 = (_var78 + _var79)
_var80 = (data_path + file_train_2)
_var81 = None
train_rfc_entropy_0322 = pd.read_csv(_var80, header=_var81)
_var82 = (data_path + file_test_mean_2)
_var83 = None
test_rfc_entropy_mean_0322 = pd.read_csv(_var82, header=_var83)
_var84 = (data_path + file_test_gmean_2)
_var85 = None
test_rfc_entropy_gmean_0322 = pd.read_csv(_var84, header=_var85)
_var86 = train_rfc_entropy_0322.shape
_var87 = 1
n_column_2 = _var86[_var87]
total_col_3 = (total_col_2 + n_column_2)
_var88 = [('rfc_entropy_0322_' + x) for x in names[:n_column]]
train_rfc_entropy_0322_0 = set_field_wrapper(train_rfc_entropy_0322, 'columns', _var88)
_var89 = [('rfc_entropy_0322_' + x) for x in names[:n_column]]
test_rfc_entropy_mean_0322_0 = set_field_wrapper(test_rfc_entropy_mean_0322, 'columns', _var89)
_var90 = [('rfc_entropy_0322_' + x) for x in names[:n_column]]
test_rfc_entropy_gmean_0322_0 = set_field_wrapper(test_rfc_entropy_gmean_0322, 'columns', _var90)
_var91 = '\ntrain_rfc_entropy: {}\t test_rfc_entropy_mean:{}\t test_rfc_entropy_gmean:{}'
_var92 = train_rfc_gini_0322_0.shape
_var93 = test_rfc_gini_mean_0322_0.shape
_var94 = test_rfc_gini_gmean_0322_0.shape
_var95 = _var91.format(_var92, _var93, _var94)
print(_var95)
_var96 = '\ntrain_rfc_entropy: {}\t test_rfc_entropy_mean:{}\t test_rfc_entropy_gmean:{}'
_var97 = train_rfc_entropy_0322_0.shape
_var98 = test_rfc_entropy_mean_0322_0.shape
_var99 = test_rfc_entropy_gmean_0322_0.shape
_var100 = _var96.format(_var97, _var98, _var99)
print(_var100)
_var101 = '\ntrain_rfc_gini_0322'
print(_var101)
_var102 = train_rfc_gini_0322_0.iloc
_var103 = 5
_var104 = 3
_var105 = _var102[:_var103, :_var104]
print(_var105)
_var106 = '\ntrain_rfc_entropy_0322'
print(_var106)
_var107 = train_rfc_entropy_0322_0.iloc
_var108 = 5
_var109 = 3
_var110 = _var107[:_var108, :_var109]
print(_var110)
_var111 = 'train_blend_LR_BM_2017-03-09-02-38'
_var112 = '.csv'
file_train_3 = (_var111 + _var112)
_var113 = 'test_blend_LR_mean_BM_2017-03-09-02-38'
_var114 = '.csv'
file_test_mean_3 = (_var113 + _var114)
_var115 = 'test_blend_LR_gmean_BM_2017-03-09-02-38'
_var116 = '.csv'
file_test_gmean_3 = (_var115 + _var116)
_var117 = (data_path + file_train_3)
_var118 = None
train_LR = pd.read_csv(_var117, header=_var118)
_var119 = (data_path + file_test_mean_3)
_var120 = None
test_LR_mean = pd.read_csv(_var119, header=_var120)
_var121 = (data_path + file_test_gmean_3)
_var122 = None
test_LR_gmean = pd.read_csv(_var121, header=_var122)
_var123 = train_LR.shape
_var124 = 1
n_column_3 = _var123[_var124]
total_col_4 = (total_col_3 + n_column_3)
_var125 = [('LR_' + x) for x in names[:n_column]]
train_LR_0 = set_field_wrapper(train_LR, 'columns', _var125)
_var126 = [('LR_' + x) for x in names[:n_column]]
test_LR_mean_0 = set_field_wrapper(test_LR_mean, 'columns', _var126)
_var127 = [('LR_' + x) for x in names[:n_column]]
test_LR_gmean_0 = set_field_wrapper(test_LR_gmean, 'columns', _var127)
_var128 = 'train_LR: {}\t test_LR_mean:{}\t test_LR_gmean:{}'
_var129 = train_LR_0.shape
_var130 = test_LR_mean_0.shape
_var131 = test_LR_gmean_0.shape
_var132 = _var128.format(_var129, _var130, _var131)
print(_var132)
_var133 = '\ntrain_LR'
print(_var133)
_var134 = train_LR_0.iloc
_var135 = 5
_var136 = 3
_var137 = _var134[:_var135, :_var136]
print(_var137)
_var138 = 'train_blend_LR_BM_0322_2017-03-22-23-38'
_var139 = '.csv'
file_train_4 = (_var138 + _var139)
_var140 = 'test_blend_LR_mean_BM_0322_2017-03-22-23-38'
_var141 = '.csv'
file_test_mean_4 = (_var140 + _var141)
_var142 = 'test_blend_LR_gmean_BM_0322_2017-03-22-23-38'
_var143 = '.csv'
file_test_gmean_4 = (_var142 + _var143)
_var144 = (data_path + file_train_4)
_var145 = None
train_LR_0322 = pd.read_csv(_var144, header=_var145)
_var146 = (data_path + file_test_mean_4)
_var147 = None
test_LR_mean_0322 = pd.read_csv(_var146, header=_var147)
_var148 = (data_path + file_test_gmean_4)
_var149 = None
test_LR_gmean_0322 = pd.read_csv(_var148, header=_var149)
_var150 = train_LR_0322.shape
_var151 = 1
n_column_4 = _var150[_var151]
total_col_5 = (total_col_4 + n_column_4)
_var152 = [('LR_0322_' + x) for x in names[:n_column]]
train_LR_0322_0 = set_field_wrapper(train_LR_0322, 'columns', _var152)
_var153 = [('LR_0322_' + x) for x in names[:n_column]]
test_LR_mean_0322_0 = set_field_wrapper(test_LR_mean_0322, 'columns', _var153)
_var154 = [('LR_0322_' + x) for x in names[:n_column]]
test_LR_gmean_0322_0 = set_field_wrapper(test_LR_gmean_0322, 'columns', _var154)
_var155 = 'train_LR_0322: {}\t test_LR_mean_0322:{}\t test_LR_gmean_0322:{}'
_var156 = train_LR_0322_0.shape
_var157 = test_LR_mean_0322_0.shape
_var158 = test_LR_gmean_0322_0.shape
_var159 = _var155.format(_var156, _var157, _var158)
print(_var159)
_var160 = '\ntrain_LR_0322'
print(_var160)
_var161 = train_LR_0322_0.iloc
_var162 = 5
_var163 = 3
_var164 = _var161[:_var162, :_var163]
print(_var164)
_var165 = 'train_blend_ET_gini_BM_2017-03-10-09-42'
_var166 = '.csv'
file_train_5 = (_var165 + _var166)
_var167 = 'test_blend_ET_gini_mean_BM_2017-03-10-09-42'
_var168 = '.csv'
file_test_mean_5 = (_var167 + _var168)
_var169 = 'test_blend_ET_gini_gmean_BM_2017-03-10-09-42'
_var170 = '.csv'
file_test_gmean_5 = (_var169 + _var170)
_var171 = (data_path + file_train_5)
_var172 = None
train_ET_gini = pd.read_csv(_var171, header=_var172)
_var173 = (data_path + file_test_mean_5)
_var174 = None
test_ET_gini_mean = pd.read_csv(_var173, header=_var174)
_var175 = (data_path + file_test_gmean_5)
_var176 = None
test_ET_gini_gmean = pd.read_csv(_var175, header=_var176)
_var177 = train_ET_gini.shape
_var178 = 1
n_column_5 = _var177[_var178]
total_col_6 = (total_col_5 + n_column_5)
_var179 = [('ET_gini_' + x) for x in names[:n_column]]
train_ET_gini_0 = set_field_wrapper(train_ET_gini, 'columns', _var179)
_var180 = [('ET_gini_' + x) for x in names[:n_column]]
test_ET_gini_mean_0 = set_field_wrapper(test_ET_gini_mean, 'columns', _var180)
_var181 = [('ET_gini_' + x) for x in names[:n_column]]
test_ET_gini_gmean_0 = set_field_wrapper(test_ET_gini_gmean, 'columns', _var181)
_var182 = 'train_blend_ET_entropy_BM_2017-03-09-20-44'
_var183 = '.csv'
file_train_6 = (_var182 + _var183)
_var184 = 'test_blend_ET_entropy_mean_BM_2017-03-09-20-44'
_var185 = '.csv'
file_test_mean_6 = (_var184 + _var185)
_var186 = 'test_blend_ET_entropy_gmean_BM_2017-03-09-20-44'
_var187 = '.csv'
file_test_gmean_6 = (_var186 + _var187)
_var188 = (data_path + file_train_6)
_var189 = None
train_ET_entropy = pd.read_csv(_var188, header=_var189)
_var190 = (data_path + file_test_mean_6)
_var191 = None
test_ET_entropy_mean = pd.read_csv(_var190, header=_var191)
_var192 = (data_path + file_test_gmean_6)
_var193 = None
test_ET_entropy_gmean = pd.read_csv(_var192, header=_var193)
_var194 = train_ET_entropy.shape
_var195 = 1
n_column_6 = _var194[_var195]
total_col_7 = (total_col_6 + n_column_6)
_var196 = [('ET_entropy_' + x) for x in names[:n_column]]
train_ET_entropy_0 = set_field_wrapper(train_ET_entropy, 'columns', _var196)
_var197 = [('ET_entropy_' + x) for x in names[:n_column]]
test_ET_entropy_mean_0 = set_field_wrapper(test_ET_entropy_mean, 'columns', _var197)
_var198 = [('ET_entropy_' + x) for x in names[:n_column]]
test_ET_entropy_gmean_0 = set_field_wrapper(test_ET_entropy_gmean, 'columns', _var198)
_var199 = 'train_ET_gini: {}\t test_ET_gini_mean:{}\t test_ET_gini_gmean:{}'
_var200 = train_ET_gini_0.shape
_var201 = test_ET_gini_mean_0.shape
_var202 = test_ET_gini_gmean_0.shape
_var203 = _var199.format(_var200, _var201, _var202)
print(_var203)
_var204 = '\ntrain_ET_entropy: {}\t test_ET_entropy_mean:{}\t test_ET_entropy_gmean:{}'
_var205 = train_ET_entropy_0.shape
_var206 = test_ET_entropy_mean_0.shape
_var207 = test_ET_entropy_gmean_0.shape
_var208 = _var204.format(_var205, _var206, _var207)
print(_var208)
_var209 = '\ntrain_ET_gini'
print(_var209)
_var210 = train_ET_gini_0.iloc
_var211 = 5
_var212 = 3
_var213 = _var210[:_var211, :_var212]
print(_var213)
_var214 = '\ntrain_ET_entropy'
print(_var214)
_var215 = train_ET_entropy_0.iloc
_var216 = 5
_var217 = 3
_var218 = _var215[:_var216, :_var217]
print(_var218)
_var219 = 'train_blend_ET_gini_BM_0322_2017-03-23-16-04'
_var220 = '.csv'
file_train_7 = (_var219 + _var220)
_var221 = 'test_blend_ET_gini_mean_BM_0322_2017-03-23-16-04'
_var222 = '.csv'
file_test_mean_7 = (_var221 + _var222)
_var223 = 'test_blend_ET_gini_gmean_BM_0322_2017-03-23-16-04'
_var224 = '.csv'
file_test_gmean_7 = (_var223 + _var224)
_var225 = (data_path + file_train_7)
_var226 = None
train_ET_gini_0322 = pd.read_csv(_var225, header=_var226)
_var227 = (data_path + file_test_mean_7)
_var228 = None
test_ET_gini_mean_0322 = pd.read_csv(_var227, header=_var228)
_var229 = (data_path + file_test_gmean_7)
_var230 = None
test_ET_gini_gmean_0322 = pd.read_csv(_var229, header=_var230)
_var231 = train_ET_gini_0322.shape
_var232 = 1
n_column_7 = _var231[_var232]
total_col_8 = (total_col_7 + n_column_7)
_var233 = [('ET_gini_0322_' + x) for x in names[:n_column]]
train_ET_gini_0322_0 = set_field_wrapper(train_ET_gini_0322, 'columns', _var233)
_var234 = [('ET_gini_0322_' + x) for x in names[:n_column]]
test_ET_gini_mean_0322_0 = set_field_wrapper(test_ET_gini_mean_0322, 'columns', _var234)
_var235 = [('ET_gini_0322_' + x) for x in names[:n_column]]
test_ET_gini_gmean_0322_0 = set_field_wrapper(test_ET_gini_gmean_0322, 'columns', _var235)
_var236 = 'train_blend_ET_entropy_BM_0322_2017-03-23-13-40'
_var237 = '.csv'
file_train_8 = (_var236 + _var237)
_var238 = 'test_blend_ET_entropy_mean_BM_0322_2017-03-23-13-40'
_var239 = '.csv'
file_test_mean_8 = (_var238 + _var239)
_var240 = 'test_blend_ET_entropy_gmean_BM_0322_2017-03-23-13-40'
_var241 = '.csv'
file_test_gmean_8 = (_var240 + _var241)
_var242 = (data_path + file_train_8)
_var243 = None
train_ET_entropy_0322 = pd.read_csv(_var242, header=_var243)
_var244 = (data_path + file_test_mean_8)
_var245 = None
test_ET_entropy_mean_0322 = pd.read_csv(_var244, header=_var245)
_var246 = (data_path + file_test_gmean_8)
_var247 = None
test_ET_entropy_gmean_0322 = pd.read_csv(_var246, header=_var247)
_var248 = train_ET_entropy_0322.shape
_var249 = 1
n_column_8 = _var248[_var249]
total_col_9 = (total_col_8 + n_column_8)
_var250 = [('ET_entropy_0322_' + x) for x in names[:n_column]]
train_ET_entropy_0322_0 = set_field_wrapper(train_ET_entropy_0322, 'columns', _var250)
_var251 = [('ET_entropy_0322_' + x) for x in names[:n_column]]
test_ET_entropy_mean_0322_0 = set_field_wrapper(test_ET_entropy_mean_0322, 'columns', _var251)
_var252 = [('ET_entropy_0322_' + x) for x in names[:n_column]]
test_ET_entropy_gmean_0322_0 = set_field_wrapper(test_ET_entropy_gmean_0322, 'columns', _var252)
_var253 = 'train_ET_gini_0322: {}\t test_ET_gini_mean_0322:{}\t test_ET_gini_gmean_0322:{}'
_var254 = train_ET_gini_0322_0.shape
_var255 = test_ET_gini_mean_0322_0.shape
_var256 = test_ET_gini_gmean_0322_0.shape
_var257 = _var253.format(_var254, _var255, _var256)
print(_var257)
_var258 = '\ntrain_ET_entropy_0322: {}\t test_ET_entropy_mean_0322:{}\t test_ET_entropy_gmean_0322:{}'
_var259 = train_ET_entropy_0322_0.shape
_var260 = test_ET_entropy_mean_0322_0.shape
_var261 = test_ET_entropy_gmean_0322_0.shape
_var262 = _var258.format(_var259, _var260, _var261)
print(_var262)
_var263 = '\ntrain_ET_gini_0322'
print(_var263)
_var264 = train_ET_gini_0322_0.iloc
_var265 = 5
_var266 = 3
_var267 = _var264[:_var265, :_var266]
print(_var267)
_var268 = '\ntrain_ET_entropy_0322'
print(_var268)
_var269 = train_ET_entropy_0322_0.iloc
_var270 = 5
_var271 = 3
_var272 = _var269[:_var270, :_var271]
print(_var272)
_var273 = 'train_blend_KNN_uniform_BM_MB_add03052240_2017-03-11-18-31'
_var274 = '.csv'
file_train_9 = (_var273 + _var274)
_var275 = 'test_blend_KNN_uniform_mean_BM_MB_add03052240_2017-03-11-18-31'
_var276 = '.csv'
file_test_mean_9 = (_var275 + _var276)
_var277 = 'test_blend_KNN_uniform_gmean_BM_MB_add03052240_2017-03-11-18-31'
_var278 = '.csv'
file_test_gmean_9 = (_var277 + _var278)
_var279 = (data_path + file_train_9)
_var280 = None
train_KNN_uniform = pd.read_csv(_var279, header=_var280)
_var281 = (data_path + file_test_mean_9)
_var282 = None
test_KNN_uniform_mean = pd.read_csv(_var281, header=_var282)
_var283 = (data_path + file_test_gmean_9)
_var284 = None
test_KNN_uniform_gmean = pd.read_csv(_var283, header=_var284)
_var285 = train_KNN_uniform.shape
_var286 = 1
n_column_9 = _var285[_var286]
total_col_10 = (total_col_9 + n_column_9)
_var287 = [('KNN_uniform_' + x) for x in names[:n_column]]
train_KNN_uniform_0 = set_field_wrapper(train_KNN_uniform, 'columns', _var287)
_var288 = [('KNN_uniform_' + x) for x in names[:n_column]]
test_KNN_uniform_mean_0 = set_field_wrapper(test_KNN_uniform_mean, 'columns', _var288)
_var289 = [('KNN_uniform_' + x) for x in names[:n_column]]
test_KNN_uniform_gmean_0 = set_field_wrapper(test_KNN_uniform_gmean, 'columns', _var289)
_var290 = 'train_blend_KNN_distance_BM_MB_add_2017-03-11-21-51'
_var291 = '.csv'
file_train_10 = (_var290 + _var291)
_var292 = 'test_blend_KNN_distance_mean_BM_MB_add_2017-03-11-21-51'
_var293 = '.csv'
file_test_mean_10 = (_var292 + _var293)
_var294 = 'test_blend_KNN_distance_gmean_BM_MB_add_2017-03-11-21-51'
_var295 = '.csv'
file_test_gmean_10 = (_var294 + _var295)
_var296 = (data_path + file_train_10)
_var297 = None
train_KNN_distance = pd.read_csv(_var296, header=_var297)
_var298 = (data_path + file_test_mean_10)
_var299 = None
test_KNN_distance_mean = pd.read_csv(_var298, header=_var299)
_var300 = (data_path + file_test_gmean_10)
_var301 = None
test_KNN_distance_gmean = pd.read_csv(_var300, header=_var301)
_var302 = train_KNN_distance.shape
_var303 = 1
n_column_10 = _var302[_var303]
total_col_11 = (total_col_10 + n_column_10)
_var304 = [('KNN_distance_' + x) for x in names[:n_column]]
train_KNN_distance_0 = set_field_wrapper(train_KNN_distance, 'columns', _var304)
_var305 = [('KNN_distance_' + x) for x in names[:n_column]]
test_KNN_distance_mean_0 = set_field_wrapper(test_KNN_distance_mean, 'columns', _var305)
_var306 = [('KNN_distance_' + x) for x in names[:n_column]]
test_KNN_distance_gmean_0 = set_field_wrapper(test_KNN_distance_gmean, 'columns', _var306)
_var307 = 'train_KNN_uniform: {}\t test_KNN_uniform_mean:{}\t test_KNN_uniform_gmean:{}'
_var308 = train_KNN_uniform_0.shape
_var309 = test_KNN_uniform_mean_0.shape
_var310 = test_KNN_uniform_gmean_0.shape
_var311 = _var307.format(_var308, _var309, _var310)
print(_var311)
_var312 = '\ntrain_KNN_distance: {}\t test_KNN_distance_mean:{}\t test_KNN_distance_gmean:{}'
_var313 = train_KNN_distance_0.shape
_var314 = test_KNN_distance_mean_0.shape
_var315 = test_KNN_distance_gmean_0.shape
_var316 = _var312.format(_var313, _var314, _var315)
print(_var316)
_var317 = '\ntrain_KNN_uniform'
print(_var317)
_var318 = train_KNN_uniform_0.iloc
_var319 = 5
_var320 = 3
_var321 = _var318[:_var319, :_var320]
print(_var321)
_var322 = '\ntrain_KNN_distance'
print(_var322)
_var323 = train_KNN_distance_0.iloc
_var324 = 5
_var325 = 3
_var326 = _var323[:_var324, :_var325]
print(_var326)
_var327 = 'train_blend_KNN_uniform_BM_0322_2017-03-24-07-31'
_var328 = '.csv'
file_train_11 = (_var327 + _var328)
_var329 = 'test_blend_KNN_uniform_mean_BM_0322_2017-03-24-07-31'
_var330 = '.csv'
file_test_mean_11 = (_var329 + _var330)
_var331 = 'test_blend_KNN_uniform_gmean_BM_0322_2017-03-24-07-31'
_var332 = '.csv'
file_test_gmean_11 = (_var331 + _var332)
_var333 = (data_path + file_train_11)
_var334 = None
train_KNN_uniform_0322 = pd.read_csv(_var333, header=_var334)
_var335 = (data_path + file_test_mean_11)
_var336 = None
test_KNN_uniform_mean_0322 = pd.read_csv(_var335, header=_var336)
_var337 = (data_path + file_test_gmean_11)
_var338 = None
test_KNN_uniform_gmean_0322 = pd.read_csv(_var337, header=_var338)
_var339 = train_KNN_uniform_0322.shape
_var340 = 1
n_column_11 = _var339[_var340]
total_col_12 = (total_col_11 + n_column_11)
_var341 = [('KNN_uniform_0322_' + x) for x in names[:n_column]]
train_KNN_uniform_0322_0 = set_field_wrapper(train_KNN_uniform_0322, 'columns', _var341)
_var342 = [('KNN_uniform_0322_' + x) for x in names[:n_column]]
test_KNN_uniform_mean_0322_0 = set_field_wrapper(test_KNN_uniform_mean_0322, 'columns', _var342)
_var343 = [('KNN_uniform_0322_' + x) for x in names[:n_column]]
test_KNN_uniform_gmean_0322_0 = set_field_wrapper(test_KNN_uniform_gmean_0322, 'columns', _var343)
_var344 = 'train_blend_KNN_distance_BM_0322_2017-03-25-08-17'
_var345 = '.csv'
file_train_12 = (_var344 + _var345)
_var346 = 'test_blend_KNN_distance_mean_BM_0322_2017-03-25-08-17'
_var347 = '.csv'
file_test_mean_12 = (_var346 + _var347)
_var348 = 'test_blend_KNN_distance_gmean_BM_0322_2017-03-25-08-17'
_var349 = '.csv'
file_test_gmean_12 = (_var348 + _var349)
_var350 = (data_path + file_train_12)
_var351 = None
train_KNN_distance_0322 = pd.read_csv(_var350, header=_var351)
_var352 = (data_path + file_test_mean_12)
_var353 = None
test_KNN_distance_mean_0322 = pd.read_csv(_var352, header=_var353)
_var354 = (data_path + file_test_gmean_12)
_var355 = None
test_KNN_distance_gmean_0322 = pd.read_csv(_var354, header=_var355)
_var356 = train_KNN_distance_0322.shape
_var357 = 1
n_column_12 = _var356[_var357]
total_col_13 = (total_col_12 + n_column_12)
_var358 = [('KNN_distance_0322_' + x) for x in names[:n_column]]
train_KNN_distance_0322_0 = set_field_wrapper(train_KNN_distance_0322, 'columns', _var358)
_var359 = [('KNN_distance_0322_' + x) for x in names[:n_column]]
test_KNN_distance_mean_0322_0 = set_field_wrapper(test_KNN_distance_mean_0322, 'columns', _var359)
_var360 = [('KNN_distance_0322_' + x) for x in names[:n_column]]
test_KNN_distance_gmean_0322_0 = set_field_wrapper(test_KNN_distance_gmean_0322, 'columns', _var360)
_var361 = 'train_KNN_uniform_0322: {}\t test_KNN_uniform_mean_0322:{}\t test_KNN_uniform_gmean_0322:{}'
_var362 = train_KNN_uniform_0322_0.shape
_var363 = test_KNN_uniform_mean_0322_0.shape
_var364 = test_KNN_uniform_gmean_0322_0.shape
_var365 = _var361.format(_var362, _var363, _var364)
print(_var365)
_var366 = '\ntrain_KNN_distance: {}\t test_KNN_distance_mean_0322:{}\t test_KNN_distance_gmean_0322:{}'
_var367 = train_KNN_distance_0322_0.shape
_var368 = test_KNN_distance_mean_0322_0.shape
_var369 = test_KNN_distance_gmean_0322_0.shape
_var370 = _var366.format(_var367, _var368, _var369)
print(_var370)
_var371 = '\ntrain_KNN_uniform_0322'
print(_var371)
_var372 = train_KNN_uniform_0322_0.iloc
_var373 = 5
_var374 = 3
_var375 = _var372[:_var373, :_var374]
print(_var375)
_var376 = '\ntrain_KNN_distance_0322'
print(_var376)
_var377 = train_KNN_distance_0322_0.iloc
_var378 = 5
_var379 = 3
_var380 = _var377[:_var378, :_var379]
print(_var380)
_var381 = 'train_blend_FM_BM_MB_add_desc_2017-03-16-09-52'
_var382 = '.csv'
file_train_13 = (_var381 + _var382)
_var383 = 'test_blend_FM_mean_BM_MB_add_desc_2017-03-16-09-52'
_var384 = '.csv'
file_test_mean_13 = (_var383 + _var384)
_var385 = 'test_blend_FM_gmean_BM_MB_add_desc_2017-03-16-09-52'
_var386 = '.csv'
file_test_gmean_13 = (_var385 + _var386)
_var387 = (data_path + file_train_13)
_var388 = None
train_FM = pd.read_csv(_var387, header=_var388)
_var389 = (data_path + file_test_mean_13)
_var390 = None
test_FM_mean = pd.read_csv(_var389, header=_var390)
_var391 = (data_path + file_test_gmean_13)
_var392 = None
test_FM_gmean = pd.read_csv(_var391, header=_var392)
_var393 = train_FM.shape
_var394 = 1
n_column_13 = _var393[_var394]
total_col_14 = (total_col_13 + n_column_13)
_var395 = [('FM_' + x) for x in names[:n_column]]
train_FM_0 = set_field_wrapper(train_FM, 'columns', _var395)
_var396 = [('FM_' + x) for x in names[:n_column]]
test_FM_mean_0 = set_field_wrapper(test_FM_mean, 'columns', _var396)
_var397 = [('FM_' + x) for x in names[:n_column]]
test_FM_gmean_0 = set_field_wrapper(test_FM_gmean, 'columns', _var397)
_var398 = 'train_FM: {}\t test_FM_mean:{}\t test_FM_gmean:{}'
_var399 = train_FM_0.shape
_var400 = test_FM_mean_0.shape
_var401 = test_FM_gmean_0.shape
_var402 = _var398.format(_var399, _var400, _var401)
print(_var402)
_var403 = '\ntrain_FM'
print(_var403)
_var404 = train_FM_0.iloc
_var405 = 5
_var406 = 3
_var407 = _var404[:_var405, :_var406]
print(_var407)
_var408 = 'train_blend_FM_BM_0322_2017-03-27-04-35'
_var409 = '.csv'
file_train_14 = (_var408 + _var409)
_var410 = 'test_blend_FM_mean_BM_0322_2017-03-27-04-35'
_var411 = '.csv'
file_test_mean_14 = (_var410 + _var411)
_var412 = 'test_blend_FM_gmean_BM_0322_2017-03-27-04-35'
_var413 = '.csv'
file_test_gmean_14 = (_var412 + _var413)
_var414 = (data_path + file_train_14)
_var415 = None
train_FM_0322 = pd.read_csv(_var414, header=_var415)
_var416 = (data_path + file_test_mean_14)
_var417 = None
test_FM_mean_0322 = pd.read_csv(_var416, header=_var417)
_var418 = (data_path + file_test_gmean_14)
_var419 = None
test_FM_gmean_0322 = pd.read_csv(_var418, header=_var419)
_var420 = train_FM_0322.shape
_var421 = 1
n_column_14 = _var420[_var421]
total_col_15 = (total_col_14 + n_column_14)
_var422 = [('FM_0322_' + x) for x in names[:n_column]]
train_FM_0322_0 = set_field_wrapper(train_FM_0322, 'columns', _var422)
_var423 = [('FM_0322_' + x) for x in names[:n_column]]
test_FM_mean_0322_0 = set_field_wrapper(test_FM_mean_0322, 'columns', _var423)
_var424 = [('FM_0322_' + x) for x in names[:n_column]]
test_FM_gmean_0322_0 = set_field_wrapper(test_FM_gmean_0322, 'columns', _var424)
_var425 = 'train_FM_0322: {}\t test_FM_mean_0322:{}\t test_FM_gmean_0322:{}'
_var426 = train_FM_0322_0.shape
_var427 = test_FM_mean_0322_0.shape
_var428 = test_FM_gmean_0322_0.shape
_var429 = _var425.format(_var426, _var427, _var428)
print(_var429)
_var430 = '\ntrain_FM_0322'
print(_var430)
_var431 = train_FM_0322_0.iloc
_var432 = 5
_var433 = 3
_var434 = _var431[:_var432, :_var433]
print(_var434)
_var435 = 'train_blend_MNB_BM_MB_add03052240_2017-03-13-20-51'
_var436 = '.csv'
file_train_15 = (_var435 + _var436)
_var437 = 'test_blend_MNB_mean_BM_MB_add03052240_2017-03-13-20-51'
_var438 = '.csv'
file_test_mean_15 = (_var437 + _var438)
_var439 = 'test_blend_MNB_gmean_BM_MB_add03052240_2017-03-13-20-51'
_var440 = '.csv'
file_test_gmean_15 = (_var439 + _var440)
_var441 = (data_path + file_train_15)
_var442 = None
train_MNB = pd.read_csv(_var441, header=_var442)
_var443 = (data_path + file_test_mean_15)
_var444 = None
test_MNB_mean = pd.read_csv(_var443, header=_var444)
_var445 = (data_path + file_test_gmean_15)
_var446 = None
test_MNB_gmean = pd.read_csv(_var445, header=_var446)
_var447 = train_MNB.shape
_var448 = 1
n_column_15 = _var447[_var448]
total_col_16 = (total_col_15 + n_column_15)
_var449 = [('MNB_' + x) for x in names[:n_column]]
train_MNB_0 = set_field_wrapper(train_MNB, 'columns', _var449)
_var450 = [('MNB_' + x) for x in names[:n_column]]
test_MNB_mean_0 = set_field_wrapper(test_MNB_mean, 'columns', _var450)
_var451 = [('MNB_' + x) for x in names[:n_column]]
test_MNB_gmean_0 = set_field_wrapper(test_MNB_gmean, 'columns', _var451)
_var452 = 'train_MNB: {}\t test_MNB_mean:{}\t test_MNB_gmean:{}'
_var453 = train_MNB_0.shape
_var454 = test_MNB_mean_0.shape
_var455 = test_MNB_gmean_0.shape
_var456 = _var452.format(_var453, _var454, _var455)
print(_var456)
_var457 = '\ntrain_MNB'
print(_var457)
_var458 = train_MNB_0.iloc
_var459 = 5
_var460 = 3
_var461 = _var458[:_var459, :_var460]
print(_var461)
_var462 = 'X_train_tsne_BM_MB_add_desc_2017-03-18-17-14'
_var463 = '.csv'
file_train_16 = (_var462 + _var463)
_var464 = 'X_test_tsne_BM_MB_add_desc_2017-03-18-17-14'
_var465 = '.csv'
file_test = (_var464 + _var465)
_var466 = (data_path + file_train_16)
_var467 = None
train_tsne = pd.read_csv(_var466, header=_var467)
_var468 = (data_path + file_test)
_var469 = None
test_tsne = pd.read_csv(_var468, header=_var469)
_var470 = train_tsne.shape
_var471 = 1
n_column_16 = _var470[_var471]
total_col_17 = (total_col_16 + n_column_16)
_var472 = ['tsne_0', 'tsne_1', 'tsne_2']
train_tsne_0 = set_field_wrapper(train_tsne, 'columns', _var472)
_var473 = ['tsne_0', 'tsne_1', 'tsne_2']
test_tsne_0 = set_field_wrapper(test_tsne, 'columns', _var473)
_var474 = 'train_tsne: {}\t test_tsne:{}'
_var475 = train_tsne_0.shape
_var476 = test_tsne_0.shape
_var477 = _var474.format(_var475, _var476)
print(_var477)
_var478 = '\ntrain_tsne'
print(_var478)
_var479 = train_tsne_0.iloc
_var480 = 5
_var481 = 3
_var482 = _var479[:_var480, :_var481]
print(_var482)
_var483 = 'X_train_tsne_BM_0322_2017-03-26-16-33'
_var484 = '.csv'
file_train_17 = (_var483 + _var484)
_var485 = 'X_test_tsne_BM_0322_2017-03-26-16-33'
_var486 = '.csv'
file_test_0 = (_var485 + _var486)
_var487 = (data_path + file_train_17)
_var488 = None
train_tsne_0322 = pd.read_csv(_var487, header=_var488)
_var489 = (data_path + file_test_0)
_var490 = None
test_tsne_0322 = pd.read_csv(_var489, header=_var490)
_var491 = train_tsne_0322.shape
_var492 = 1
n_column_17 = _var491[_var492]
total_col_18 = (total_col_17 + n_column_17)
_var493 = ['tsne_0_0322', 'tsne_1_0322', 'tsne_2_0322']
train_tsne_0322_0 = set_field_wrapper(train_tsne_0322, 'columns', _var493)
_var494 = ['tsne_0_0322', 'tsne_1_0322', 'tsne_2_0322']
test_tsne_0322_0 = set_field_wrapper(test_tsne_0322, 'columns', _var494)
_var495 = 'train_tsne_0322: {}\t test_tsne_0322:{}'
_var496 = train_tsne_0322_0.shape
_var497 = test_tsne_0322_0.shape
_var498 = _var495.format(_var496, _var497)
print(_var498)
_var499 = '\ntrain_tsne_0322'
print(_var499)
_var500 = train_tsne_0322_0.iloc
_var501 = 5
_var502 = 3
_var503 = _var500[:_var501, :_var502]
print(_var503)
_var504 = 'train_blend_xgb_BM_MB_add_desc_2017-03-14-16-54'
_var505 = '.csv'
file_train_18 = (_var504 + _var505)
_var506 = 'test_blend_xgb_mean_BM_MB_add_desc_2017-03-14-16-54'
_var507 = '.csv'
file_test_mean_16 = (_var506 + _var507)
_var508 = 'test_blend_xgb_gmean_BM_MB_add_desc_2017-03-14-16-54'
_var509 = '.csv'
file_test_gmean_16 = (_var508 + _var509)
_var510 = (data_path + file_train_18)
_var511 = None
train_xgb = pd.read_csv(_var510, header=_var511)
_var512 = (data_path + file_test_mean_16)
_var513 = None
test_xgb_mean = pd.read_csv(_var512, header=_var513)
_var514 = (data_path + file_test_gmean_16)
_var515 = None
test_xgb_gmean = pd.read_csv(_var514, header=_var515)
_var516 = train_xgb.shape
_var517 = 1
n_column_18 = _var516[_var517]
total_col_19 = (total_col_18 + n_column_18)
_var518 = [('xgb_' + x) for x in names[:n_column]]
train_xgb_0 = set_field_wrapper(train_xgb, 'columns', _var518)
_var519 = [('xgb_' + x) for x in names[:n_column]]
test_xgb_mean_0 = set_field_wrapper(test_xgb_mean, 'columns', _var519)
_var520 = [('xgb_' + x) for x in names[:n_column]]
test_xgb_gmean_0 = set_field_wrapper(test_xgb_gmean, 'columns', _var520)
_var521 = 'train_xgb: {}\t test_xgb_mean:{}\t test_xgb_gmean:{}'
_var522 = train_xgb_0.shape
_var523 = test_xgb_mean_0.shape
_var524 = test_xgb_gmean_0.shape
_var525 = _var521.format(_var522, _var523, _var524)
print(_var525)
_var526 = '\ntrain_xgb'
print(_var526)
_var527 = train_xgb_0.iloc
_var528 = 5
_var529 = 3
_var530 = _var527[:_var528, :_var529]
print(_var530)
_var531 = 'train_blend_xgb_BM_0322_2017-03-25-19-12'
_var532 = '.csv'
file_train_19 = (_var531 + _var532)
_var533 = 'test_blend_xgb_mean_BM_0322_2017-03-25-19-12'
_var534 = '.csv'
file_test_mean_17 = (_var533 + _var534)
_var535 = 'test_blend_xgb_gmean_BM_0322_2017-03-25-19-12'
_var536 = '.csv'
file_test_gmean_17 = (_var535 + _var536)
_var537 = (data_path + file_train_19)
_var538 = None
train_xgb_0322 = pd.read_csv(_var537, header=_var538)
_var539 = (data_path + file_test_mean_17)
_var540 = None
test_xgb_mean_0322 = pd.read_csv(_var539, header=_var540)
_var541 = (data_path + file_test_gmean_17)
_var542 = None
test_xgb_gmean_0322 = pd.read_csv(_var541, header=_var542)
_var543 = train_xgb_0322.shape
_var544 = 1
n_column_19 = _var543[_var544]
total_col_20 = (total_col_19 + n_column_19)
_var545 = [('xgb_0322_' + x) for x in names[:n_column]]
train_xgb_0322_0 = set_field_wrapper(train_xgb_0322, 'columns', _var545)
_var546 = [('xgb_0322_' + x) for x in names[:n_column]]
test_xgb_mean_0322_0 = set_field_wrapper(test_xgb_mean_0322, 'columns', _var546)
_var547 = [('xgb_0322_' + x) for x in names[:n_column]]
test_xgb_gmean_0322_0 = set_field_wrapper(test_xgb_gmean_0322, 'columns', _var547)
_var548 = 'train_xgb_0322: {}\t test_xgb_mean_0322:{}\t test_xgb_gmean_0322:{}'
_var549 = train_xgb_0322_0.shape
_var550 = test_xgb_mean_0322_0.shape
_var551 = test_xgb_gmean_0322_0.shape
_var552 = _var548.format(_var549, _var550, _var551)
print(_var552)
_var553 = '\ntrain_xgb_0322'
print(_var553)
_var554 = train_xgb_0322_0.iloc
_var555 = 5
_var556 = 3
_var557 = _var554[:_var555, :_var556]
print(_var557)
_var558 = 'train_blend_xgb_BM_0331_2017-04-02-17-55'
_var559 = '.csv'
file_train_20 = (_var558 + _var559)
_var560 = 'test_blend_xgb_mean_BM_0331_2017-04-02-17-55'
_var561 = '.csv'
file_test_mean_18 = (_var560 + _var561)
_var562 = 'test_blend_xgb_gmean_BM_0331_2017-04-02-17-55'
_var563 = '.csv'
file_test_gmean_18 = (_var562 + _var563)
_var564 = (data_path + file_train_20)
_var565 = None
train_xgb_0331 = pd.read_csv(_var564, header=_var565)
_var566 = (data_path + file_test_mean_18)
_var567 = None
test_xgb_mean_0331 = pd.read_csv(_var566, header=_var567)
_var568 = (data_path + file_test_gmean_18)
_var569 = None
test_xgb_gmean_0331 = pd.read_csv(_var568, header=_var569)
_var570 = train_xgb_0331.shape
_var571 = 1
n_column_20 = _var570[_var571]
total_col_21 = (total_col_20 + n_column_20)
_var572 = [('xgb_0331_' + x) for x in names[:n_column]]
train_xgb_0331_0 = set_field_wrapper(train_xgb_0331, 'columns', _var572)
_var573 = [('xgb_0331_' + x) for x in names[:n_column]]
test_xgb_mean_0331_0 = set_field_wrapper(test_xgb_mean_0331, 'columns', _var573)
_var574 = [('xgb_0331_' + x) for x in names[:n_column]]
test_xgb_gmean_0331_0 = set_field_wrapper(test_xgb_gmean_0331, 'columns', _var574)
_var575 = 'train_xgb_0331: {}\t test_xgb_mean_0331:{}\t test_xgb_gmean_0331:{}'
_var576 = train_xgb_0331_0.shape
_var577 = test_xgb_mean_0331_0.shape
_var578 = test_xgb_gmean_0331_0.shape
_var579 = _var575.format(_var576, _var577, _var578)
print(_var579)
_var580 = '\ntrain_xgb_0331'
print(_var580)
_var581 = train_xgb_0331_0.iloc
_var582 = 5
_var583 = 3
_var584 = _var581[:_var582, :_var583]
print(_var584)
_var585 = 'train_blend_xgb_BM_0331_30blend_2017-04-04-09-15'
_var586 = '.csv'
file_train_21 = (_var585 + _var586)
_var587 = 'test_blend_xgb_mean_BM_0331_30blend_2017-04-04-09-15'
_var588 = '.csv'
file_test_mean_19 = (_var587 + _var588)
_var589 = 'test_blend_xgb_gmean_BM_0331_30blend_2017-04-04-09-15'
_var590 = '.csv'
file_test_gmean_19 = (_var589 + _var590)
_var591 = (data_path + file_train_21)
_var592 = None
train_xgb_0331_30fold = pd.read_csv(_var591, header=_var592)
_var593 = (data_path + file_test_mean_19)
_var594 = None
test_xgb_mean_0331_30fold = pd.read_csv(_var593, header=_var594)
_var595 = (data_path + file_test_gmean_19)
_var596 = None
test_xgb_gmean_0331_30fold = pd.read_csv(_var595, header=_var596)
_var597 = train_xgb_0331_30fold.shape
_var598 = 1
n_column_21 = _var597[_var598]
total_col_22 = (total_col_21 + n_column_21)
_var599 = [('xgb_0331_30fold_' + x) for x in names[:n_column]]
train_xgb_0331_30fold_0 = set_field_wrapper(train_xgb_0331_30fold, 'columns', _var599)
_var600 = [('xgb_0331_30fold_' + x) for x in names[:n_column]]
test_xgb_mean_0331_30fold_0 = set_field_wrapper(test_xgb_mean_0331_30fold, 'columns', _var600)
_var601 = [('xgb_0331_30fold_' + x) for x in names[:n_column]]
test_xgb_gmean_0331_30fold_0 = set_field_wrapper(test_xgb_gmean_0331_30fold, 'columns', _var601)
_var602 = 'train_xgb_0331_30fold: {}\t test_xgb_mean_0331_30fold:{}\t test_xgb_gmean_0331_30fold:{}'
_var603 = train_xgb_0331_30fold_0.shape
_var604 = test_xgb_mean_0331_30fold_0.shape
_var605 = test_xgb_gmean_0331_30fold_0.shape
_var606 = _var602.format(_var603, _var604, _var605)
print(_var606)
_var607 = '\ntrain_xgb_0331_30fold'
print(_var607)
_var608 = train_xgb_0331_30fold_0.iloc
_var609 = 5
_var610 = 3
_var611 = _var608[:_var609, :_var610]
print(_var611)
_var612 = 'train_blend_xgb_cv137_BM_2017-04-06-11-44'
_var613 = '.csv'
file_train_22 = (_var612 + _var613)
_var614 = 'test_blend_xgb_mean_cv137_5blend_BM_2017-04-06-11-44'
_var615 = '.csv'
file_test_mean_20 = (_var614 + _var615)
_var616 = 'test_blend_xgb_gmean_cv137_5blend_BM_2017-04-06-11-44'
_var617 = '.csv'
file_test_gmean_20 = (_var616 + _var617)
_var618 = (data_path + file_train_22)
_var619 = None
train_xgb_cv137 = pd.read_csv(_var618, header=_var619)
_var620 = (data_path + file_test_mean_20)
_var621 = None
test_xgb_mean_cv137 = pd.read_csv(_var620, header=_var621)
_var622 = (data_path + file_test_gmean_20)
_var623 = None
test_xgb_gmean_cv137 = pd.read_csv(_var622, header=_var623)
_var624 = train_xgb_cv137.shape
_var625 = 1
n_column_22 = _var624[_var625]
total_col_23 = (total_col_22 + n_column_22)
_var626 = [('xgb_cv137_' + x) for x in names[:n_column]]
train_xgb_cv137_0 = set_field_wrapper(train_xgb_cv137, 'columns', _var626)
_var627 = [('xgb_cv137_' + x) for x in names[:n_column]]
test_xgb_mean_cv137_0 = set_field_wrapper(test_xgb_mean_cv137, 'columns', _var627)
_var628 = [('xgb_cv137_' + x) for x in names[:n_column]]
test_xgb_gmean_cv137_0 = set_field_wrapper(test_xgb_gmean_cv137, 'columns', _var628)
_var629 = 'train_xgb_cv137: {}\t test_xgb_mean_cv137:{}\t test_xgb_gmean_cv137:{}'
_var630 = train_xgb_cv137_0.shape
_var631 = test_xgb_mean_cv137_0.shape
_var632 = test_xgb_gmean_cv137_0.shape
_var633 = _var629.format(_var630, _var631, _var632)
print(_var633)
_var634 = '\ntrain_xgb_cv137'
print(_var634)
_var635 = train_xgb_cv137_0.iloc
_var636 = 5
_var637 = 3
_var638 = _var635[:_var636, :_var637]
print(_var638)
_var639 = 'train_blend_xgb_cv137_BM_2017-04-06-15-28'
_var640 = '.csv'
file_train_23 = (_var639 + _var640)
_var641 = 'test_blend_xgb_mean_cv137_5blend_BM_2017-04-06-15-28'
_var642 = '.csv'
file_test_mean_21 = (_var641 + _var642)
_var643 = 'test_blend_xgb_gmean_cv137_5blend_BM_2017-04-06-15-28'
_var644 = '.csv'
file_test_gmean_21 = (_var643 + _var644)
_var645 = (data_path + file_train_23)
_var646 = None
train_xgb_cv137_1 = pd.read_csv(_var645, header=_var646)
_var647 = (data_path + file_test_mean_21)
_var648 = None
test_xgb_mean_cv137_1 = pd.read_csv(_var647, header=_var648)
_var649 = (data_path + file_test_gmean_21)
_var650 = None
test_xgb_gmean_cv137_1 = pd.read_csv(_var649, header=_var650)
_var651 = train_xgb_cv137_1.shape
_var652 = 1
n_column_23 = _var651[_var652]
total_col_24 = (total_col_23 + n_column_23)
_var653 = [('xgb_cv137_1_' + x) for x in names[:n_column]]
train_xgb_cv137_1_0 = set_field_wrapper(train_xgb_cv137_1, 'columns', _var653)
_var654 = [('xgb_cv137_1_' + x) for x in names[:n_column]]
test_xgb_mean_cv137_1_0 = set_field_wrapper(test_xgb_mean_cv137_1, 'columns', _var654)
_var655 = [('xgb_cv137_1_' + x) for x in names[:n_column]]
test_xgb_gmean_cv137_1_0 = set_field_wrapper(test_xgb_gmean_cv137_1, 'columns', _var655)
_var656 = 'train_xgb_cv137_1: {}\t test_xgb_mean_cv137_1:{}\t test_xgb_gmean_cv137_1:{}'
_var657 = train_xgb_cv137_1_0.shape
_var658 = test_xgb_mean_cv137_1_0.shape
_var659 = test_xgb_gmean_cv137_1_0.shape
_var660 = _var656.format(_var657, _var658, _var659)
print(_var660)
_var661 = '\ntrain_xgb_cv137_1'
print(_var661)
_var662 = train_xgb_cv137_1_0.iloc
_var663 = 5
_var664 = 3
_var665 = _var662[:_var663, :_var664]
print(_var665)
_var666 = 'train_blend_xgb_cv_price_BM_2017-04-09-14-06'
_var667 = '.csv'
file_train_24 = (_var666 + _var667)
_var668 = 'test_blend_xgb_mean_cv_price_BM_2017-04-09-14-06'
_var669 = '.csv'
file_test_mean_22 = (_var668 + _var669)
_var670 = 'test_blend_xgb_gmean_cv_price_BM_2017-04-09-14-06'
_var671 = '.csv'
file_test_gmean_22 = (_var670 + _var671)
_var672 = (data_path + file_train_24)
_var673 = None
train_xgb_cv_price = pd.read_csv(_var672, header=_var673)
_var674 = (data_path + file_test_mean_22)
_var675 = None
test_xgb_mean_cv_price = pd.read_csv(_var674, header=_var675)
_var676 = (data_path + file_test_gmean_22)
_var677 = None
test_xgb_gmean_cv_price = pd.read_csv(_var676, header=_var677)
_var678 = train_xgb_cv_price.shape
_var679 = 1
n_column_24 = _var678[_var679]
total_col_25 = (total_col_24 + n_column_24)
_var680 = [('xgb_cv_price_' + x) for x in names[:n_column]]
train_xgb_cv_price_0 = set_field_wrapper(train_xgb_cv_price, 'columns', _var680)
_var681 = [('xgb_cv_price_' + x) for x in names[:n_column]]
test_xgb_mean_cv_price_0 = set_field_wrapper(test_xgb_mean_cv_price, 'columns', _var681)
_var682 = [('xgb_cv_price_' + x) for x in names[:n_column]]
test_xgb_gmean_cv_price_0 = set_field_wrapper(test_xgb_gmean_cv_price, 'columns', _var682)
_var683 = 'train_xgb_cv_price: {}\t test_xgb_mean_cv_price:{}\t test_xgb_gmean_cv_price:{}'
_var684 = train_xgb_cv_price_0.shape
_var685 = test_xgb_mean_cv_price_0.shape
_var686 = test_xgb_gmean_cv_price_0.shape
_var687 = _var683.format(_var684, _var685, _var686)
print(_var687)
_var688 = '\ntrain_xgb_cv_price'
print(_var688)
_var689 = train_xgb_cv_price_0.iloc
_var690 = 5
_var691 = 3
_var692 = _var689[:_var690, :_var691]
print(_var692)
_var693 = 'train_blend_xgb_CV_MS_BM_2017-04-11-09-18'
_var694 = '.csv'
file_train_25 = (_var693 + _var694)
_var695 = 'test_blend_xgb_mean_CV_MS_BM_2017-04-11-09-18'
_var696 = '.csv'
file_test_mean_23 = (_var695 + _var696)
_var697 = 'test_blend_xgb_gmean_CV_MS_BM_2017-04-11-09-18'
_var698 = '.csv'
file_test_gmean_23 = (_var697 + _var698)
_var699 = (data_path + file_train_25)
_var700 = None
train_xgb_cv_MS_52571 = pd.read_csv(_var699, header=_var700)
_var701 = (data_path + file_test_mean_23)
_var702 = None
test_xgb_mean_cv_MS_52571 = pd.read_csv(_var701, header=_var702)
_var703 = (data_path + file_test_gmean_23)
_var704 = None
test_xgb_gmean_cv_MS_52571 = pd.read_csv(_var703, header=_var704)
_var705 = train_xgb_cv_MS_52571.shape
_var706 = 1
n_column_25 = _var705[_var706]
total_col_26 = (total_col_25 + n_column_25)
_var707 = [('xgb_cv_MS_52571_' + x) for x in names[:n_column]]
train_xgb_cv_MS_52571_0 = set_field_wrapper(train_xgb_cv_MS_52571, 'columns', _var707)
_var708 = [('xgb_cv_MS_52571_' + x) for x in names[:n_column]]
test_xgb_mean_cv_MS_52571_0 = set_field_wrapper(test_xgb_mean_cv_MS_52571, 'columns', _var708)
_var709 = [('xgb_cv_MS_52571_' + x) for x in names[:n_column]]
test_xgb_gmean_cv_MS_52571_0 = set_field_wrapper(test_xgb_gmean_cv_MS_52571, 'columns', _var709)
_var710 = 'train_xgb_cv_MS_52571: {}\t test_xgb_mean_cv_MS_52571:{}\t test_xgb_gmean_cv_MS_52571:{}'
_var711 = train_xgb_cv_MS_52571_0.shape
_var712 = test_xgb_mean_cv_MS_52571_0.shape
_var713 = test_xgb_gmean_cv_MS_52571_0.shape
_var714 = _var710.format(_var711, _var712, _var713)
print(_var714)
_var715 = '\ntrain_xgb_cv_MS_52571'
print(_var715)
_var716 = train_xgb_cv_MS_52571_0.iloc
_var717 = 5
_var718 = 3
_var719 = _var716[:_var717, :_var718]
print(_var719)
_var720 = 'train_blend_xgb_CV_MS_30blend_BM_2017-04-12-08-56'
_var721 = '.csv'
file_train_26 = (_var720 + _var721)
_var722 = 'test_blend_xgb_mean_CV_MS_30blend_BM_2017-04-12-08-56'
_var723 = '.csv'
file_test_mean_24 = (_var722 + _var723)
_var724 = 'test_blend_xgb_gmean_CV_MS_30blend_BM_2017-04-12-08-56'
_var725 = '.csv'
file_test_gmean_24 = (_var724 + _var725)
_var726 = (data_path + file_train_26)
_var727 = None
train_xgb_cv_MS_52571_30fold = pd.read_csv(_var726, header=_var727)
_var728 = (data_path + file_test_mean_24)
_var729 = None
test_xgb_mean_cv_MS_52571_30fold = pd.read_csv(_var728, header=_var729)
_var730 = (data_path + file_test_gmean_24)
_var731 = None
test_xgb_gmean_cv_MS_52571_30fold = pd.read_csv(_var730, header=_var731)
_var732 = train_xgb_cv_MS_52571_30fold.shape
_var733 = 1
n_column_26 = _var732[_var733]
total_col_27 = (total_col_26 + n_column_26)
_var734 = [('xgb_cv_MS_52571_30fold_' + x) for x in names[:n_column]]
train_xgb_cv_MS_52571_30fold_0 = set_field_wrapper(train_xgb_cv_MS_52571_30fold, 'columns', _var734)
_var735 = [('xgb_cv_MS_52571_30fold_' + x) for x in names[:n_column]]
test_xgb_mean_cv_MS_52571_30fold_0 = set_field_wrapper(test_xgb_mean_cv_MS_52571_30fold, 'columns', _var735)
_var736 = [('xgb_cv_MS_52571_30fold_' + x) for x in names[:n_column]]
test_xgb_gmean_cv_MS_52571_30fold_0 = set_field_wrapper(test_xgb_gmean_cv_MS_52571_30fold, 'columns', _var736)
_var737 = 'train_xgb_cv_MS_52571_30fold: {}\t test_xgb_mean_cv_MS_52571_30fold:{}\t test_xgb_gmean_cv_MS_52571_30fold:{}'
_var738 = train_xgb_cv_MS_52571_30fold_0.shape
_var739 = test_xgb_mean_cv_MS_52571_30fold_0.shape
_var740 = test_xgb_gmean_cv_MS_52571_30fold_0.shape
_var741 = _var737.format(_var738, _var739, _var740)
print(_var741)
_var742 = '\ntrain_xgb_cv_MS_52571_30fold'
print(_var742)
_var743 = train_xgb_cv_MS_52571_30fold_0.iloc
_var744 = 5
_var745 = 3
_var746 = _var743[:_var744, :_var745]
print(_var746)
_var747 = 'train_blend_xgb_ovr_BM_0322_2017-03-27-19-36'
_var748 = '.csv'
file_train_27 = (_var747 + _var748)
_var749 = 'test_blend_xgb_ovr_mean_BM_0322_2017-03-27-19-36'
_var750 = '.csv'
file_test_mean_25 = (_var749 + _var750)
_var751 = 'test_blend_xgb_ovr_gmean_BM_0322_2017-03-27-19-36'
_var752 = '.csv'
file_test_gmean_25 = (_var751 + _var752)
_var753 = (data_path + file_train_27)
_var754 = None
train_xgb_ovr_0322 = pd.read_csv(_var753, header=_var754)
_var755 = (data_path + file_test_mean_25)
_var756 = None
test_xgb_mean_ovr_0322 = pd.read_csv(_var755, header=_var756)
_var757 = (data_path + file_test_gmean_25)
_var758 = None
test_xgb_gmean_ovr_0322 = pd.read_csv(_var757, header=_var758)
_var759 = train_xgb_ovr_0322.shape
_var760 = 1
n_column_27 = _var759[_var760]
total_col_28 = (total_col_27 + n_column_27)
_var761 = [('xgb_0322_ovr_' + x) for x in names[:n_column]]
train_xgb_ovr_0322_0 = set_field_wrapper(train_xgb_ovr_0322, 'columns', _var761)
_var762 = [('xgb_0322_ovr_' + x) for x in names[:n_column]]
test_xgb_mean_ovr_0322_0 = set_field_wrapper(test_xgb_mean_ovr_0322, 'columns', _var762)
_var763 = [('xgb_0322_ovr_' + x) for x in names[:n_column]]
test_xgb_gmean_ovr_0322_0 = set_field_wrapper(test_xgb_gmean_ovr_0322, 'columns', _var763)
_var764 = 'train_xgb_0322: {}\t test_xgb_mean_0322:{}\t test_xgb_gmean_0322:{}'
_var765 = train_xgb_ovr_0322_0.shape
_var766 = test_xgb_mean_ovr_0322_0.shape
_var767 = test_xgb_gmean_ovr_0322_0.shape
_var768 = _var764.format(_var765, _var766, _var767)
print(_var768)
_var769 = '\ntrain_xgb_ovr_0322'
print(_var769)
_var770 = train_xgb_ovr_0322_0.iloc
_var771 = 5
_var772 = 3
_var773 = _var770[:_var771, :_var772]
print(_var773)
_var774 = 'train_blend_LightGBM_BM_0322_2017-03-27-08-21'
_var775 = '.csv'
file_train_28 = (_var774 + _var775)
_var776 = 'test_blend_LightGBM_mean_BM_0322_2017-03-27-08-21'
_var777 = '.csv'
file_test_mean_26 = (_var776 + _var777)
_var778 = 'test_blend_LightGBM_gmean_BM_0322_2017-03-27-08-21'
_var779 = '.csv'
file_test_gmean_26 = (_var778 + _var779)
_var780 = (data_path + file_train_28)
_var781 = None
train_lgb_0322 = pd.read_csv(_var780, header=_var781)
_var782 = (data_path + file_test_mean_26)
_var783 = None
test_lgb_mean_0322 = pd.read_csv(_var782, header=_var783)
_var784 = (data_path + file_test_gmean_26)
_var785 = None
test_lgb_gmean_0322 = pd.read_csv(_var784, header=_var785)
_var786 = train_lgb_0322.shape
_var787 = 1
n_column_28 = _var786[_var787]
total_col_29 = (total_col_28 + n_column_28)
_var788 = [('lgb_0322_' + x) for x in names[:n_column]]
train_lgb_0322_0 = set_field_wrapper(train_lgb_0322, 'columns', _var788)
_var789 = [('lgb_0322_' + x) for x in names[:n_column]]
test_lgb_mean_0322_0 = set_field_wrapper(test_lgb_mean_0322, 'columns', _var789)
_var790 = [('lgb_0322_' + x) for x in names[:n_column]]
test_lgb_gmean_0322_0 = set_field_wrapper(test_lgb_gmean_0322, 'columns', _var790)
_var791 = 'train_lgb_0322: {}\t test_lgb_mean_0322:{}\t test_lgb_gmean_0322:{}'
_var792 = train_lgb_0322_0.shape
_var793 = test_lgb_mean_0322_0.shape
_var794 = test_lgb_gmean_0322_0.shape
_var795 = _var791.format(_var792, _var793, _var794)
print(_var795)
_var796 = '\ntrain_lgb_0322'
print(_var796)
_var797 = train_lgb_0322_0.iloc
_var798 = 5
_var799 = 3
_var800 = _var797[:_var798, :_var799]
print(_var800)
_var801 = 'train_blend_LightGBM_dart_BM_0322_2017-03-31-13-03'
_var802 = '.csv'
file_train_29 = (_var801 + _var802)
_var803 = 'test_blend_LightGBM_dart_mean_BM_0322_2017-03-31-13-03'
_var804 = '.csv'
file_test_mean_27 = (_var803 + _var804)
_var805 = 'test_blend_LightGBM_dart_gmean_BM_0322_2017-03-31-13-03'
_var806 = '.csv'
file_test_gmean_27 = (_var805 + _var806)
_var807 = (data_path + file_train_29)
_var808 = None
train_lgb_dart_0322 = pd.read_csv(_var807, header=_var808)
_var809 = (data_path + file_test_mean_27)
_var810 = None
test_lgb_mean_dart_0322 = pd.read_csv(_var809, header=_var810)
_var811 = (data_path + file_test_gmean_27)
_var812 = None
test_lgb_gmean_dart_0322 = pd.read_csv(_var811, header=_var812)
_var813 = train_lgb_dart_0322.shape
_var814 = 1
n_column_29 = _var813[_var814]
total_col_30 = (total_col_29 + n_column_29)
_var815 = [('lgb_dart_0322_' + x) for x in names[:n_column]]
train_lgb_dart_0322_0 = set_field_wrapper(train_lgb_dart_0322, 'columns', _var815)
_var816 = [('lgb_dart_0322_' + x) for x in names[:n_column]]
test_lgb_mean_dart_0322_0 = set_field_wrapper(test_lgb_mean_dart_0322, 'columns', _var816)
_var817 = [('lgb_dart_0322_' + x) for x in names[:n_column]]
test_lgb_gmean_dart_0322_0 = set_field_wrapper(test_lgb_gmean_dart_0322, 'columns', _var817)
_var818 = 'train_lgb_dart_0322: {}\t test_lgb_mean_dart_0322:{}\t test_lgb_gmean_dart_0322:{}'
_var819 = train_lgb_dart_0322_0.shape
_var820 = test_lgb_mean_dart_0322_0.shape
_var821 = test_lgb_gmean_dart_0322_0.shape
_var822 = _var818.format(_var819, _var820, _var821)
print(_var822)
_var823 = '\ntrain_lgb_dart_0322'
print(_var823)
_var824 = train_lgb_dart_0322_0.iloc
_var825 = 5
_var826 = 3
_var827 = _var824[:_var825, :_var826]
print(_var827)
_var828 = 'train_blend_LightGBM_BM_0331_2017-04-01-07-33'
_var829 = '.csv'
file_train_30 = (_var828 + _var829)
_var830 = 'test_blend_LightGBM_mean_BM_0331_2017-04-01-07-33'
_var831 = '.csv'
file_test_mean_28 = (_var830 + _var831)
_var832 = 'test_blend_LightGBM_gmean_BM_0331_2017-04-01-07-33'
_var833 = '.csv'
file_test_gmean_28 = (_var832 + _var833)
_var834 = (data_path + file_train_30)
_var835 = None
train_lgb_0331 = pd.read_csv(_var834, header=_var835)
_var836 = (data_path + file_test_mean_28)
_var837 = None
test_lgb_mean_0331 = pd.read_csv(_var836, header=_var837)
_var838 = (data_path + file_test_gmean_28)
_var839 = None
test_lgb_gmean_0331 = pd.read_csv(_var838, header=_var839)
_var840 = train_lgb_0331.shape
_var841 = 1
n_column_30 = _var840[_var841]
total_col_31 = (total_col_30 + n_column_30)
_var842 = [('lgb_0331_' + x) for x in names[:n_column]]
train_lgb_0331_0 = set_field_wrapper(train_lgb_0331, 'columns', _var842)
_var843 = [('lgb_0331_' + x) for x in names[:n_column]]
test_lgb_mean_0331_0 = set_field_wrapper(test_lgb_mean_0331, 'columns', _var843)
_var844 = [('lgb_0331_' + x) for x in names[:n_column]]
test_lgb_gmean_0331_0 = set_field_wrapper(test_lgb_gmean_0331, 'columns', _var844)
_var845 = 'train_lgb_0331: {}\t test_lgb_mean_0331:{}\t test_lgb_gmean_0331:{}'
_var846 = train_lgb_0331_0.shape
_var847 = test_lgb_mean_0331_0.shape
_var848 = test_lgb_gmean_0331_0.shape
_var849 = _var845.format(_var846, _var847, _var848)
print(_var849)
_var850 = '\ntrain_lgb_0331'
print(_var850)
_var851 = train_lgb_0331_0.iloc
_var852 = 5
_var853 = 3
_var854 = _var851[:_var852, :_var853]
print(_var854)
_var855 = 'train_blend_LightGBM_BM_0401_2017-04-02-12-24'
_var856 = '.csv'
file_train_31 = (_var855 + _var856)
_var857 = 'test_blend_LightGBM_mean_BM_0401_2017-04-02-12-24'
_var858 = '.csv'
file_test_mean_29 = (_var857 + _var858)
_var859 = 'test_blend_LightGBM_gmean_BM_0401_2017-04-02-12-24'
_var860 = '.csv'
file_test_gmean_29 = (_var859 + _var860)
_var861 = (data_path + file_train_31)
_var862 = None
train_lgb_0401 = pd.read_csv(_var861, header=_var862)
_var863 = (data_path + file_test_mean_29)
_var864 = None
test_lgb_mean_0401 = pd.read_csv(_var863, header=_var864)
_var865 = (data_path + file_test_gmean_29)
_var866 = None
test_lgb_gmean_0401 = pd.read_csv(_var865, header=_var866)
_var867 = train_lgb_0401.shape
_var868 = 1
n_column_31 = _var867[_var868]
total_col_32 = (total_col_31 + n_column_31)
_var869 = [('lgb_0401_' + x) for x in names[:n_column]]
train_lgb_0401_0 = set_field_wrapper(train_lgb_0401, 'columns', _var869)
_var870 = [('lgb_0401_' + x) for x in names[:n_column]]
test_lgb_mean_0401_0 = set_field_wrapper(test_lgb_mean_0401, 'columns', _var870)
_var871 = [('lgb_0401_' + x) for x in names[:n_column]]
test_lgb_gmean_0401_0 = set_field_wrapper(test_lgb_gmean_0401, 'columns', _var871)
_var872 = 'train_lgb_0401: {}\t test_lgb_mean_0401:{}\t test_lgb_gmean_0401:{}'
_var873 = train_lgb_0401_0.shape
_var874 = test_lgb_mean_0401_0.shape
_var875 = test_lgb_gmean_0401_0.shape
_var876 = _var872.format(_var873, _var874, _var875)
print(_var876)
_var877 = '\ntrain_lgb_0401'
print(_var877)
_var878 = train_lgb_0401_0.iloc
_var879 = 5
_var880 = 3
_var881 = _var878[:_var879, :_var880]
print(_var881)
_var882 = 'train_blend_Keras_BM_0331_2017-04-04-15-32'
_var883 = '.csv'
file_train_32 = (_var882 + _var883)
_var884 = 'test_blend_Keras_BM_0331_2017-04-04-15-32'
_var885 = '.csv'
file_test_mean_30 = (_var884 + _var885)
_var886 = (data_path + file_train_32)
_var887 = None
train_nn_0331 = pd.read_csv(_var886, header=_var887)
_var888 = (data_path + file_test_mean_30)
_var889 = None
test_nn_mean_0331 = pd.read_csv(_var888, header=_var889)
_var890 = train_nn_0331.shape
_var891 = 1
n_column_32 = _var890[_var891]
total_col_33 = (total_col_32 + n_column_32)
_var892 = [('nn_0331_' + x) for x in names[:n_column]]
train_nn_0331_0 = set_field_wrapper(train_nn_0331, 'columns', _var892)
_var893 = [('nn_0331_' + x) for x in names[:n_column]]
test_nn_mean_0331_0 = set_field_wrapper(test_nn_mean_0331, 'columns', _var893)
_var894 = 'train_nn_0331: {}\t test_nn_mean_0331:{}'
_var895 = train_nn_0331_0.shape
_var896 = test_nn_mean_0331_0.shape
_var897 = _var894.format(_var895, _var896)
print(_var897)
_var898 = '\ntrain_nn_0331'
print(_var898)
_var899 = train_nn_0331_0.iloc
_var900 = 5
_var901 = 3
_var902 = _var899[:_var900, :_var901]
print(_var902)
_var903 = 'train_blend_Keras_BM_0331_2017-04-04-17-23'
_var904 = '.csv'
file_train_33 = (_var903 + _var904)
_var905 = 'test_blend_Keras_BM_0331_2017-04-04-17-23'
_var906 = '.csv'
file_test_mean_31 = (_var905 + _var906)
_var907 = (data_path + file_train_33)
_var908 = None
train_nn_0331_1 = pd.read_csv(_var907, header=_var908)
_var909 = (data_path + file_test_mean_31)
_var910 = None
test_nn_mean_0331_1 = pd.read_csv(_var909, header=_var910)
_var911 = train_nn_0331_1.shape
_var912 = 1
n_column_33 = _var911[_var912]
total_col_34 = (total_col_33 + n_column_33)
_var913 = [('nn_0331_1_' + x) for x in names[:n_column]]
train_nn_0331_1_0 = set_field_wrapper(train_nn_0331_1, 'columns', _var913)
_var914 = [('nn_0331_1_' + x) for x in names[:n_column]]
test_nn_mean_0331_1_0 = set_field_wrapper(test_nn_mean_0331_1, 'columns', _var914)
_var915 = 'train_nn_0331_1: {}\t test_nn_mean_0331_1:{}'
_var916 = train_nn_0331_1_0.shape
_var917 = test_nn_mean_0331_1_0.shape
_var918 = _var915.format(_var916, _var917)
print(_var918)
_var919 = '\ntrain_nn_0331_1'
print(_var919)
_var920 = train_nn_0331_1_0.iloc
_var921 = 5
_var922 = 3
_var923 = _var920[:_var921, :_var922]
print(_var923)
_var924 = 'train_blend_Keras_ovr_BM_0331_2017-04-05-03-37'
_var925 = '.csv'
file_train_34 = (_var924 + _var925)
_var926 = 'test_blend_Keras_ovr_BM_0331_2017-04-05-03-37'
_var927 = '.csv'
file_test_mean_32 = (_var926 + _var927)
_var928 = (data_path + file_train_34)
_var929 = None
train_nn_ovr_0331 = pd.read_csv(_var928, header=_var929)
_var930 = (data_path + file_test_mean_32)
_var931 = None
test_nn_mean_ovr_0331 = pd.read_csv(_var930, header=_var931)
_var932 = train_nn_ovr_0331.shape
_var933 = 1
n_column_34 = _var932[_var933]
total_col_35 = (total_col_34 + n_column_34)
_var934 = [('nn_0331_ovr_' + x) for x in names[:n_column]]
train_nn_ovr_0331_0 = set_field_wrapper(train_nn_ovr_0331, 'columns', _var934)
_var935 = [('nn_0331_ovr_' + x) for x in names[:n_column]]
test_nn_mean_ovr_0331_0 = set_field_wrapper(test_nn_mean_ovr_0331, 'columns', _var935)
_var936 = 'train_nn_ovr_0331: {}\t test_nn_mean_ovr_0331:{}'
_var937 = train_nn_ovr_0331_0.shape
_var938 = test_nn_mean_ovr_0331_0.shape
_var939 = _var936.format(_var937, _var938)
print(_var939)
_var940 = '\ntrain_nn_ovr_0331'
print(_var940)
_var941 = train_nn_ovr_0331_0.iloc
_var942 = 5
_var943 = 3
_var944 = _var941[:_var942, :_var943]
print(_var944)
_var945 = 'train_blend_Keras_CV_52571_BM_2017-04-13-13-59'
_var946 = '.csv'
file_train_35 = (_var945 + _var946)
_var947 = 'test_blend_Keras_mean_CV_52571_BM_2017-04-13-13-59'
_var948 = '.csv'
file_test_mean_33 = (_var947 + _var948)
_var949 = (data_path + file_train_35)
_var950 = None
train_nn_bagging = pd.read_csv(_var949, header=_var950)
_var951 = (data_path + file_test_mean_33)
_var952 = None
test_nn_mean_bagging = pd.read_csv(_var951, header=_var952)
_var953 = train_nn_bagging.shape
_var954 = 1
n_column_35 = _var953[_var954]
total_col_36 = (total_col_35 + n_column_35)
_var955 = [('nn_bagging_' + x) for x in names[:n_column]]
train_nn_bagging_0 = set_field_wrapper(train_nn_bagging, 'columns', _var955)
_var956 = [('nn_bagging_' + x) for x in names[:n_column]]
test_nn_mean_bagging_0 = set_field_wrapper(test_nn_mean_bagging, 'columns', _var956)
_var957 = 'train_nn_bagging: {}\t test_nn_mean_bagging:{}'
_var958 = train_nn_bagging_0.shape
_var959 = test_nn_mean_bagging_0.shape
_var960 = _var957.format(_var958, _var959)
print(_var960)
_var961 = '\ntrain_nn_bagging'
print(_var961)
_var962 = train_nn_bagging_0.iloc
_var963 = 5
_var964 = 3
_var965 = _var962[:_var963, :_var964]
print(_var965)
_var966 = 'train_blend_GP_BM_2017-04-09-19-15'
_var967 = '.csv'
file_train_36 = (_var966 + _var967)
_var968 = 'test_blend_GP_BM_2017-04-09-19-15'
_var969 = '.csv'
file_test_mean_34 = (_var968 + _var969)
_var970 = (data_path + file_train_36)
_var971 = None
train_gp = pd.read_csv(_var970, header=_var971)
_var972 = (data_path + file_test_mean_34)
_var973 = None
test_gp = pd.read_csv(_var972, header=_var973)
_var974 = train_gp.shape
_var975 = 1
n_column_36 = _var974[_var975]
total_col_37 = (total_col_36 + n_column_36)
_var976 = [('gp_' + x) for x in names[:n_column]]
train_gp_0 = set_field_wrapper(train_gp, 'columns', _var976)
_var977 = [('gp_' + x) for x in names[:n_column]]
test_gp_0 = set_field_wrapper(test_gp, 'columns', _var977)
_var978 = 'train_gp: {}\t test_gp:{}'
_var979 = train_gp_0.shape
_var980 = test_gp_0.shape
_var981 = _var978.format(_var979, _var980)
print(_var981)
_var982 = '\ntrain_gp'
print(_var982)
_var983 = train_gp_0.iloc
_var984 = 5
_var985 = 3
_var986 = _var983[:_var984, :_var985]
print(_var986)
_var987 = 'train_blend_XGB_BM_3bagging_CV_MS_52571_2017-04-13-09-33'
_var988 = '.csv'
file_train_37 = (_var987 + _var988)
_var989 = 'test_blend_XGB_BM_3bagging_CV_MS_52571_2017-04-13-09-33'
_var990 = '.csv'
file_test_mean_35 = (_var989 + _var990)
_var991 = (data_path + file_train_37)
_var992 = None
train_bagging_0 = pd.read_csv(_var991, header=_var992)
_var993 = (data_path + file_test_mean_35)
_var994 = None
test_bagging_0 = pd.read_csv(_var993, header=_var994)
_var995 = train_bagging_0.shape
_var996 = 1
n_column_37 = _var995[_var996]
total_col_38 = (total_col_37 + n_column_37)
_var997 = [('bagging_0_' + x) for x in names[:n_column]]
train_bagging_0_0 = set_field_wrapper(train_bagging_0, 'columns', _var997)
_var998 = [('bagging_0_' + x) for x in names[:n_column]]
test_bagging_0_0 = set_field_wrapper(test_bagging_0, 'columns', _var998)
_var999 = 'train_bagging_0: {}\t test_bagging_0:{}'
_var1000 = train_bagging_0_0.shape
_var1001 = test_bagging_0_0.shape
_var1002 = _var999.format(_var1000, _var1001)
print(_var1002)
_var1003 = '\ntrain_bagging_0'
print(_var1003)
_var1004 = train_bagging_0_0.iloc
_var1005 = 5
_var1006 = 3
_var1007 = _var1004[:_var1005, :_var1006]
print(_var1007)
_var1008 = 'train_blend_xgb_141bagging_BM_2017-04-13-10-12'
_var1009 = '.csv'
file_train_38 = (_var1008 + _var1009)
_var1010 = 'test_blend_xgb_mean_141bagging_BM_2017-04-13-10-12'
_var1011 = '.csv'
file_test_mean_36 = (_var1010 + _var1011)
_var1012 = (data_path + file_train_38)
_var1013 = None
train_bagging_1 = pd.read_csv(_var1012, header=_var1013)
_var1014 = (data_path + file_test_mean_36)
_var1015 = None
test_bagging_1 = pd.read_csv(_var1014, header=_var1015)
_var1016 = train_bagging_1.shape
_var1017 = 1
n_column_38 = _var1016[_var1017]
total_col_39 = (total_col_38 + n_column_38)
_var1018 = [('bagging_1_' + x) for x in names[:n_column]]
train_bagging_1_0 = set_field_wrapper(train_bagging_1, 'columns', _var1018)
_var1019 = [('bagging_1_' + x) for x in names[:n_column]]
test_bagging_1_0 = set_field_wrapper(test_bagging_1, 'columns', _var1019)
_var1020 = 'train_bagging_1: {}\t test_bagging_1:{}'
_var1021 = train_bagging_0_0.shape
_var1022 = test_bagging_0_0.shape
_var1023 = _var1020.format(_var1021, _var1022)
print(_var1023)
_var1024 = '\ntrain_bagging_1'
print(_var1024)
_var1025 = train_bagging_0_0.iloc
_var1026 = 5
_var1027 = 3
_var1028 = _var1025[:_var1026, :_var1027]
print(_var1028)
print(total_col_39)
_var1029 = [train_rfc_gini_0, train_rfc_entropy_0, train_rfc_gini_0322_0, train_rfc_entropy_0322_0, train_LR_0, train_LR_0322_0, train_ET_gini_0, train_ET_entropy_0, train_ET_gini_0322_0, train_ET_entropy_0322_0, train_KNN_uniform_0, train_KNN_distance_0, train_KNN_uniform_0322_0, train_KNN_distance_0322_0, train_FM_0, train_FM_0322_0, train_MNB_0, train_tsne_0, train_tsne_0322_0, train_xgb_0, train_xgb_0322_0, train_xgb_0331_0, train_xgb_0331_30fold_0, train_xgb_cv137_0, train_xgb_cv137_1_0, train_xgb_cv_price_0, train_xgb_cv_MS_52571_0, train_xgb_cv_MS_52571_30fold_0, train_xgb_ovr_0322_0, train_lgb_0322_0, train_lgb_dart_0322_0, train_lgb_0331_0, train_lgb_0401_0, train_nn_0331_0, train_nn_0331_1_0, train_nn_ovr_0331_0, train_nn_bagging_0, train_gp_0, train_bagging_0_0, train_bagging_1_0]
_var1030 = 1
train_2nd = pd.concat(_var1029, axis=_var1030)
_var1031 = [test_rfc_gini_mean_0, test_rfc_entropy_mean_0, test_rfc_gini_mean_0322_0, test_rfc_entropy_mean_0322_0, test_LR_mean_0, test_LR_mean_0322_0, test_ET_gini_mean_0, test_ET_entropy_mean_0, test_ET_gini_mean_0322_0, test_ET_entropy_mean_0322_0, test_KNN_uniform_mean_0, test_KNN_distance_mean_0, test_KNN_uniform_mean_0322_0, test_KNN_distance_mean_0322_0, test_FM_mean_0, test_FM_mean_0322_0, test_MNB_mean_0, test_tsne_0, test_tsne_0322_0, test_xgb_mean_0, test_xgb_mean_0322_0, test_xgb_mean_0331_0, test_xgb_mean_0331_30fold_0, test_xgb_mean_cv137_0, test_xgb_mean_cv137_1_0, test_xgb_mean_cv_price_0, test_xgb_mean_cv_MS_52571_0, test_xgb_mean_cv_MS_52571_30fold_0, test_xgb_mean_ovr_0322_0, test_lgb_mean_0322_0, test_lgb_mean_dart_0322_0, test_lgb_mean_0331_0, test_lgb_mean_0401_0, test_nn_mean_0331_0, test_nn_mean_0331_1_0, test_nn_mean_ovr_0331_0, test_nn_mean_bagging_0, test_gp_0, test_bagging_0_0, test_bagging_1_0]
_var1032 = 1
test_2nd_mean = pd.concat(_var1031, axis=_var1032)
_var1033 = [test_rfc_gini_gmean_0, test_rfc_entropy_gmean_0, test_rfc_gini_gmean_0322_0, test_rfc_entropy_gmean_0322_0, test_LR_gmean_0, test_LR_gmean_0322_0, test_ET_gini_gmean_0, test_ET_entropy_gmean_0, test_ET_gini_gmean_0322_0, test_ET_entropy_gmean_0322_0, test_KNN_uniform_gmean_0, test_KNN_distance_gmean_0, test_KNN_uniform_gmean_0322_0, test_KNN_distance_gmean_0322_0, test_FM_gmean_0, test_FM_gmean_0322_0, test_MNB_gmean_0, test_tsne_0, test_tsne_0322_0, test_xgb_gmean_0, test_xgb_gmean_0322_0, test_xgb_gmean_0331_0, test_xgb_gmean_0331_30fold_0, test_xgb_gmean_cv137_0, test_xgb_gmean_cv137_1_0, test_xgb_gmean_cv_price_0, test_xgb_gmean_cv_MS_52571_0, test_xgb_gmean_cv_MS_52571_30fold_0, test_xgb_gmean_ovr_0322_0, test_lgb_gmean_0322_0, test_lgb_gmean_dart_0322_0, test_lgb_gmean_0331_0, test_lgb_gmean_0401_0, test_nn_mean_0331_0, test_nn_mean_0331_1_0, test_nn_mean_ovr_0331_0, test_nn_mean_bagging_0, test_gp_0, test_bagging_0_0, test_bagging_1_0]
_var1034 = 1
test_2nd_gmean = pd.concat(_var1033, axis=_var1034)
_var1035 = 'train_2nd: {}\t test_2nd_mean:{}\t test_2nd_gmean:{}'
_var1036 = train_2nd.shape
_var1037 = test_2nd_mean.shape
_var1038 = test_2nd_gmean.shape
_var1039 = _var1035.format(_var1036, _var1037, _var1038)
print(_var1039)
data_path_0 = '../input/'
_var1040 = 'train_BM_0331.csv'
_var1041 = (data_path_0 + _var1040)
train_X = pd.read_csv(_var1041)
_var1042 = 'test_BM_0331.csv'
_var1043 = (data_path_0 + _var1042)
test_X = pd.read_csv(_var1043)
_var1044 = '../input/'
_var1045 = 'labels_BrandenMurray.csv'
_var1046 = (_var1044 + _var1045)
_var1047 = pd.read_csv(_var1046)
train_y = np.ravel(_var1047)
train_y_0 = to_categorical(train_y)
_var1048 = train_X.shape
_var1049 = 0
ntrain = _var1048[_var1049]
_var1050 = test_X.listing_id
_var1051 = 'int32'
_var1052 = _var1050.astype(_var1051)
sub_id = _var1052.values
_var1053 = train_X.shape
_var1054 = test_X.shape
_var1055 = train_y_0.shape
_var1056 = (_var1053, _var1054, _var1055)
print(_var1056)
_var1057 = test_X.num_loc_price_diff
null_ind = _var1057.isnull()
_var1058 = 'num_loc_price_diff'
_var1059 = 'num_price'
_var1060 = test_X[_var1059]
_var1061 = 'num_loc_median_price'
_var1062 = test_X[_var1061]
_var1063 = (_var1060 - _var1062)
test_X_0 = set_index_wrapper(test_X, _var1058, _var1063)
_var1064 = [train_X, train_2nd]
_var1065 = 1
train_X_0 = pd.concat(_var1064, axis=_var1065)
_var1066 = [test_X_0, test_2nd_mean]
_var1067 = 1
test_X_1 = pd.concat(_var1066, axis=_var1067)
_var1068 = train_X_0.shape
print(_var1068)
_var1069 = test_X_1.shape
print(_var1069)
_var1070 = [train_X_0, test_X_1]
full_data = pd.concat(_var1070)
_var1071 = full_data.shape
print(_var1071)
_var1072 = full_data.isnull()
_var1073 = _var1072.values
_var1073.any()
_var1074 = full_data.columns
_var1074.values
feat_to_use = ['building_id_mean_med', 'building_id_mean_high', 'manager_id_mean_med', 'manager_id_mean_high', 'median_price_bed', 'ratio_bed', 'compound', 'neg', 'neu', 'pos', 'listing_id', 'num_latitude', 'num_longitude', 'num_dist_from_center', 'num_OutlierAggregated', 'num_pos_density', 'num_building_null', 'num_fbuilding', 'num_fmanager', 'num_created_weekday', 'num_created_weekofyear', 'num_created_day', 'num_created_month', 'num_created_hour', 'num_bathrooms', 'num_bedrooms', 'num_price', 'num_price_q', 'num_priceXroom', 'num_even_bathrooms', 'num_features', 'num_photos', 'num_desc_length', 'num_desc_length_null', 'num_6_median_price', 'num_6_price_ratio', 'num_6_price_diff', 'num_loc_median_price', 'num_loc_price_ratio', 'num_loc_price_diff', 'num_loc_ratio', 'num_loc_diff', 'hcc_pos_pred_1', 'hcc_pos_pred_2', 'building_id', 'display_address', 'manager_id', 'street_address', 'num_pricePerBed', 'num_pricePerBath', 'num_pricePerRoom', 'num_bedPerBath', 'num_bedBathDiff', 'num_bedBathSum', 'num_bedsPerc', 'rfc_gini_low_0', 'rfc_gini_medium_0', 'rfc_gini_high_0', 'rfc_entropy_low_0', 'rfc_entropy_medium_0', 'rfc_entropy_high_0', 'rfc_gini_0322_low_0', 'rfc_gini_0322_medium_0', 'rfc_gini_0322_high_0', 'rfc_entropy_0322_low_0', 'rfc_entropy_0322_medium_0', 'rfc_entropy_0322_high_0', 'LR_low_0', 'LR_medium_0', 'LR_high_0', 'LR_low_1', 'LR_medium_1', 'LR_high_1', 'LR_low_2', 'LR_medium_2', 'LR_high_2', 'LR_low_3', 'LR_medium_3', 'LR_high_3', 'LR_low_4', 'LR_medium_4', 'LR_high_4', 'LR_low_5', 'LR_medium_5', 'LR_high_5', 'LR_low_6', 'LR_medium_6', 'LR_high_6', 'LR_0322_low_0', 'LR_0322_medium_0', 'LR_0322_high_0', 'LR_0322_low_1', 'LR_0322_medium_1', 'LR_0322_high_1', 'LR_0322_low_2', 'LR_0322_medium_2', 'LR_0322_high_2', 'LR_0322_low_3', 'LR_0322_medium_3', 'LR_0322_high_3', 'LR_0322_low_4', 'LR_0322_medium_4', 'LR_0322_high_4', 'LR_0322_low_5', 'LR_0322_medium_5', 'LR_0322_high_5', 'LR_0322_low_6', 'LR_0322_medium_6', 'LR_0322_high_6', 'ET_gini_low_0', 'ET_gini_medium_0', 'ET_gini_high_0', 'ET_entropy_low_0', 'ET_entropy_medium_0', 'ET_entropy_high_0', 'ET_gini_0322_low_0', 'ET_gini_0322_medium_0', 'ET_gini_0322_high_0', 'ET_entropy_0322_low_0', 'ET_entropy_0322_medium_0', 'ET_entropy_0322_high_0', 'KNN_uniform_low_0', 'KNN_uniform_medium_0', 'KNN_uniform_high_0', 'KNN_distance_low_0', 'KNN_distance_medium_0', 'KNN_distance_high_0', 'KNN_uniform_0322_low_0', 'KNN_uniform_0322_medium_0', 'KNN_uniform_0322_high_0', 'KNN_uniform_0322_low_1', 'KNN_uniform_0322_medium_1', 'KNN_uniform_0322_high_1', 'KNN_uniform_0322_low_2', 'KNN_uniform_0322_medium_2', 'KNN_uniform_0322_high_2', 'KNN_uniform_0322_low_3', 'KNN_uniform_0322_medium_3', 'KNN_uniform_0322_high_3', 'KNN_uniform_0322_low_4', 'KNN_uniform_0322_medium_4', 'KNN_uniform_0322_high_4', 'KNN_distance_0322_low_0', 'KNN_distance_0322_medium_0', 'KNN_distance_0322_high_0', 'KNN_distance_0322_low_1', 'KNN_distance_0322_medium_1', 'KNN_distance_0322_high_1', 'KNN_distance_0322_low_2', 'KNN_distance_0322_medium_2', 'KNN_distance_0322_high_2', 'KNN_distance_0322_low_3', 'KNN_distance_0322_medium_3', 'KNN_distance_0322_high_3', 'KNN_distance_0322_low_4', 'KNN_distance_0322_medium_4', 'KNN_distance_0322_high_4', 'FM_low_0', 'FM_medium_0', 'FM_high_0', 'FM_0322_low_0', 'FM_0322_medium_0', 'FM_0322_high_0', 'MNB_low_0', 'MNB_medium_0', 'MNB_high_0', 'MNB_low_1', 'MNB_medium_1', 'MNB_high_1', 'MNB_low_2', 'MNB_medium_2', 'MNB_high_2', 'tsne_0', 'tsne_1', 'tsne_2', 'tsne_0_0322', 'tsne_1_0322', 'tsne_2_0322', 'xgb_low_0', 'xgb_medium_0', 'xgb_high_0', 'xgb_low_1', 'xgb_medium_1', 'xgb_high_1', 'xgb_low_2', 'xgb_medium_2', 'xgb_high_2', 'xgb_low_3', 'xgb_medium_3', 'xgb_high_3', 'xgb_low_4', 'xgb_medium_4', 'xgb_high_4', 'xgb_0322_low_0', 'xgb_0322_medium_0', 'xgb_0322_high_0', 'xgb_0322_low_1', 'xgb_0322_medium_1', 'xgb_0322_high_1', 'xgb_0322_low_2', 'xgb_0322_medium_2', 'xgb_0322_high_2', 'xgb_0322_low_3', 'xgb_0322_medium_3', 'xgb_0322_high_3', 'xgb_0322_low_4', 'xgb_0322_medium_4', 'xgb_0322_high_4', 'xgb_0331_low_0', 'xgb_0331_medium_0', 'xgb_0331_high_0', 'xgb_0331_low_1', 'xgb_0331_medium_1', 'xgb_0331_high_1', 'xgb_0331_low_2', 'xgb_0331_medium_2', 'xgb_0331_high_2', 'xgb_0331_low_3', 'xgb_0331_medium_3', 'xgb_0331_high_3', 'xgb_0331_low_4', 'xgb_0331_medium_4', 'xgb_0331_high_4', 'xgb_0331_30fold_low_0', 'xgb_0331_30fold_medium_0', 'xgb_0331_30fold_high_0', 'xgb_cv137_low_0', 'xgb_cv137_medium_0', 'xgb_cv137_high_0', 'xgb_cv137_1_low_0', 'xgb_cv137_1_medium_0', 'xgb_cv137_1_high_0', 'xgb_cv_price_low_0', 'xgb_cv_price_medium_0', 'xgb_cv_price_high_0', 'xgb_cv_MS_52571_low_0', 'xgb_cv_MS_52571_medium_0', 'xgb_cv_MS_52571_high_0', 'xgb_cv_MS_52571_low_1', 'xgb_cv_MS_52571_medium_1', 'xgb_cv_MS_52571_high_1', 'xgb_cv_MS_52571_low_2', 'xgb_cv_MS_52571_medium_2', 'xgb_cv_MS_52571_high_2', 'xgb_cv_MS_52571_low_3', 'xgb_cv_MS_52571_medium_3', 'xgb_cv_MS_52571_high_3', 'xgb_cv_MS_52571_low_4', 'xgb_cv_MS_52571_medium_4', 'xgb_cv_MS_52571_high_4', 'xgb_cv_MS_52571_30fold_low_0', 'xgb_cv_MS_52571_30fold_medium_0', 'xgb_cv_MS_52571_30fold_high_0', 'xgb_0322_ovr_low_0', 'xgb_0322_ovr_medium_0', 'xgb_0322_ovr_high_0', 'lgb_0322_low_0', 'lgb_0322_medium_0', 'lgb_0322_high_0', 'lgb_0322_low_1', 'lgb_0322_medium_1', 'lgb_0322_high_1', 'lgb_0322_low_2', 'lgb_0322_medium_2', 'lgb_0322_high_2', 'lgb_0322_low_3', 'lgb_0322_medium_3', 'lgb_0322_high_3', 'lgb_0322_low_4', 'lgb_0322_medium_4', 'lgb_0322_high_4', 'lgb_dart_0322_low_0', 'lgb_dart_0322_medium_0', 'lgb_dart_0322_high_0', 'lgb_dart_0322_low_1', 'lgb_dart_0322_medium_1', 'lgb_dart_0322_high_1', 'lgb_dart_0322_low_2', 'lgb_dart_0322_medium_2', 'lgb_dart_0322_high_2', 'lgb_dart_0322_low_3', 'lgb_dart_0322_medium_3', 'lgb_dart_0322_high_3', 'lgb_dart_0322_low_4', 'lgb_dart_0322_medium_4', 'lgb_dart_0322_high_4', 'lgb_0331_low_0', 'lgb_0331_medium_0', 'lgb_0331_high_0', 'lgb_0331_low_1', 'lgb_0331_medium_1', 'lgb_0331_high_1', 'lgb_0331_low_2', 'lgb_0331_medium_2', 'lgb_0331_high_2', 'lgb_0331_low_3', 'lgb_0331_medium_3', 'lgb_0331_high_3', 'lgb_0331_low_4', 'lgb_0331_medium_4', 'lgb_0331_high_4', 'lgb_0401_low_0', 'lgb_0401_medium_0', 'lgb_0401_high_0', 'lgb_0401_low_1', 'lgb_0401_medium_1', 'lgb_0401_high_1', 'lgb_0401_low_2', 'lgb_0401_medium_2', 'lgb_0401_high_2', 'lgb_0401_low_3', 'lgb_0401_medium_3', 'lgb_0401_high_3', 'lgb_0401_low_4', 'lgb_0401_medium_4', 'lgb_0401_high_4', 'nn_0331_low_0', 'nn_0331_medium_0', 'nn_0331_high_0', 'nn_0331_low_1', 'nn_0331_medium_1', 'nn_0331_high_1', 'nn_0331_1_low_0', 'nn_0331_1_medium_0', 'nn_0331_1_high_0', 'nn_0331_1_low_1', 'nn_0331_1_medium_1', 'nn_0331_1_high_1', 'nn_0331_ovr_low_0', 'nn_0331_ovr_medium_0', 'nn_0331_ovr_high_0', 'nn_0331_ovr_low_1', 'nn_0331_ovr_medium_1', 'nn_0331_ovr_high_1', 'nn_bagging_low_0', 'nn_bagging_medium_0', 'nn_bagging_high_0', 'nn_bagging_low_1', 'nn_bagging_medium_1', 'nn_bagging_high_1', 'gp_low_0', 'gp_medium_0', 'gp_high_0', 'bagging_0_low_0', 'bagging_0_medium_0', 'bagging_0_high_0', 'bagging_1_low_0', 'bagging_1_medium_0', 'bagging_1_high_0']
for col in feat_to_use:
    _var1075 = full_data.loc
    _var1076 = preprocessing.StandardScaler()
    _var1077 = full_data[col]
    _var1078 = _var1077.values
    _var1079 = (- 1)
    _var1080 = 1
    _var1081 = _var1078.reshape(_var1079, _var1080)
    _var1082 = _var1076.fit_transform(_var1081)
    _var1075_0 = set_index_wrapper(_var1075, (slice(None, None, None), col), _var1082)
train_df_nn = full_data[:ntrain]
test_df_nn = full_data[ntrain:]
train_df_nn_0 = sparse.csr_matrix(train_df_nn)
test_df_nn_0 = sparse.csr_matrix(test_df_nn)
_var1083 = train_df_nn_0.shape
print(_var1083)
_var1084 = test_df_nn_0.shape
print(_var1084)
_var1089 = 0.8
_var1090 = 3
(_var1085, _var1086, _var1087, _var1088) = train_test_split(train_df_nn_0, train_y_0, train_size=_var1089, random_state=_var1090)
X_train = _var1085
X_val = _var1086
y_train = _var1087
y_val = _var1088

def batch_generator(X, y, batch_size, shuffle):
    _var1091 = X.shape
    _var1092 = 0
    _var1093 = _var1091[_var1092]
    _var1094 = (_var1093 / batch_size)
    number_of_batches = np.ceil(_var1094)
    counter = 0
    _var1095 = X.shape
    _var1096 = 0
    _var1097 = _var1095[_var1096]
    sample_index = np.arange(_var1097)
    if shuffle:
        _var1098 = np.random
        _var1098.shuffle(sample_index)
    _var1099 = True
    while _var1099:
        _var1100 = (batch_size * counter)
        _var1101 = 1
        _var1102 = (counter + _var1101)
        _var1103 = (batch_size * _var1102)
        batch_index = sample_index[_var1100:_var1103]
        _var1104 = X[batch_index, :]
        X_batch = _var1104.toarray()
        y_batch = y[batch_index]
        _var1105 = 1
        counter_0 = (counter + _var1105)
        (yield (X_batch, y_batch))
        _var1106 = (counter_0 == number_of_batches)
        if _var1106:
            if shuffle:
                _var1107 = np.random
                _var1107.shuffle(sample_index)
            counter_1 = 0
        counter_2 = __phi__(counter_1, counter_0)
    counter_3 = __phi__(counter_2, counter)

def batch_generatorp(X_0, batch_size_0, shuffle_0):
    _var1108 = X_0.shape
    _var1109 = 0
    _var1110 = _var1108[_var1109]
    _var1111 = X_0.shape
    _var1112 = 0
    _var1113 = _var1111[_var1112]
    _var1114 = (_var1113 / batch_size_0)
    _var1115 = np.ceil(_var1114)
    number_of_batches_0 = (_var1110 / _var1115)
    counter_4 = 0
    _var1116 = X_0.shape
    _var1117 = 0
    _var1118 = _var1116[_var1117]
    sample_index_0 = np.arange(_var1118)
    _var1119 = True
    while _var1119:
        _var1120 = (batch_size_0 * counter_4)
        _var1121 = 1
        _var1122 = (counter_4 + _var1121)
        _var1123 = (batch_size_0 * _var1122)
        batch_index_0 = sample_index_0[_var1120:_var1123]
        _var1124 = X_0[batch_index_0, :]
        X_batch_0 = _var1124.toarray()
        _var1125 = 1
        counter_5 = (counter_4 + _var1125)
        (yield X_batch_0)
        _var1126 = (counter_5 == number_of_batches_0)
        if _var1126:
            counter_6 = 0
        counter_7 = __phi__(counter_6, counter_5)
    counter_8 = __phi__(counter_7, counter_4)
_var1127 = 'val_loss'
_var1128 = 5
_var1129 = 0
early_stop = EarlyStopping(monitor=_var1127, patience=_var1128, verbose=_var1129)
_var1130 = 'weights.hdf5'
_var1131 = 'val_loss'
_var1132 = 0
_var1133 = True
checkpointer = ModelCheckpoint(filepath=_var1130, monitor=_var1131, verbose=_var1132, save_best_only=_var1133)

def create_model(input_dim):
    model = Sequential()
    init = 'glorot_uniform'
    _var1134 = 200
    _var1135 = Dense(_var1134, input_dim=input_dim, init=init)
    model.add(_var1135)
    _var1136 = 'sigmoid'
    _var1137 = Activation(_var1136)
    model.add(_var1137)
    _var1138 = PReLU()
    model.add(_var1138)
    _var1139 = BatchNormalization()
    model.add(_var1139)
    _var1140 = 0.4
    _var1141 = Dropout(_var1140)
    model.add(_var1141)
    _var1142 = 20
    _var1143 = Dense(_var1142, init=init)
    model.add(_var1143)
    _var1144 = 'sigmoid'
    _var1145 = Activation(_var1144)
    model.add(_var1145)
    _var1146 = PReLU()
    model.add(_var1146)
    _var1147 = BatchNormalization()
    model.add(_var1147)
    _var1148 = 0.4
    _var1149 = Dropout(_var1148)
    model.add(_var1149)
    _var1150 = 3
    _var1151 = 'softmax'
    _var1152 = Dense(_var1150, init=init, activation=_var1151)
    model.add(_var1152)
    _var1153 = 'categorical_crossentropy'
    _var1154 = 'Adamax'
    model.compile(loss=_var1153, optimizer=_var1154)
    return model
_var1155 = X_train.shape
_var1156 = 1
_var1157 = _var1155[_var1156]
model_0 = create_model(_var1157)
_var1158 = 256
_var1159 = True
_var1160 = batch_generator(X_train, y_train, _var1158, _var1159)
_var1161 = 1000
_var1162 = X_val.todense()
_var1163 = (_var1162, y_val)
_var1164 = [early_stop, checkpointer]
fit = model_0.fit_generator(generator=_var1160, nb_epoch=_var1161, samples_per_epoch=ntrain, validation_data=_var1163, callbacks=_var1164)
model_1 = fit
_var1165 = fit.history
_var1166 = 'val_loss'
_var1167 = _var1165[_var1166]
_var1168 = min(_var1167)
print(_var1168)
_var1169 = 'weights.hdf5'
model_1.load_weights(_var1169)
_var1170 = 'categorical_crossentropy'
_var1171 = 'adam'
model_1.compile(loss=_var1170, optimizer=_var1171)
_var1172 = test_df_nn_0.toarray()
_var1173 = 128
_var1174 = 0
pred_y = model_1.predict_proba(x=_var1172, batch_size=_var1173, verbose=_var1174)
pred_y
now = datetime.now()
_var1175 = '../output/sub_Keras_'
_var1176 = '%Y-%m-%d-%H-%M'
_var1177 = now.strftime(_var1176)
_var1178 = str(_var1177)
_var1179 = (_var1175 + _var1178)
_var1180 = '.csv'
sub_name = (_var1179 + _var1180)
out_df = pd.DataFrame(pred_y)
_var1181 = ['low', 'medium', 'high']
out_df_0 = set_field_wrapper(out_df, 'columns', _var1181)
_var1182 = 'listing_id'
out_df_1 = set_index_wrapper(out_df_0, _var1182, sub_id)
_var1183 = False
out_df_1.to_csv(sub_name, index=_var1183)

def nn_model(params):
    model_2 = Sequential()
    init_0 = 'glorot_uniform'
    _var1184 = 'input_size'
    _var1185 = params[_var1184]
    _var1186 = 'input_dim'
    _var1187 = params[_var1186]
    _var1188 = Dense(_var1185, input_dim=_var1187, init=init_0)
    model_2.add(_var1188)
    _var1189 = 'sigmoid'
    _var1190 = Activation(_var1189)
    model_2.add(_var1190)
    _var1191 = PReLU()
    model_2.add(_var1191)
    _var1192 = BatchNormalization()
    model_2.add(_var1192)
    _var1193 = 'input_drop_out'
    _var1194 = params[_var1193]
    _var1195 = Dropout(_var1194)
    model_2.add(_var1195)
    _var1196 = 'hidden_size'
    _var1197 = params[_var1196]
    _var1198 = Dense(_var1197, init=init_0)
    model_2.add(_var1198)
    _var1199 = 'sigmoid'
    _var1200 = Activation(_var1199)
    model_2.add(_var1200)
    _var1201 = PReLU()
    model_2.add(_var1201)
    _var1202 = BatchNormalization()
    model_2.add(_var1202)
    _var1203 = 'hidden_drop_out'
    _var1204 = params[_var1203]
    _var1205 = Dropout(_var1204)
    model_2.add(_var1205)
    _var1206 = 3
    _var1207 = 'softmax'
    _var1208 = Dense(_var1206, init=init_0, activation=_var1207)
    model_2.add(_var1208)
    _var1209 = 'categorical_crossentropy'
    _var1210 = 'Adamax'
    model_2.compile(loss=_var1209, optimizer=_var1210)
    return model_2

def nn_blend_data(parameters, train_x, train_y_1, test_x, fold, early_stopping_rounds=0, batch_size_1=128, randomseed=1234):
    N_params = len(parameters)
    _var1211 = True
    skf = KFold(n_splits=fold, shuffle=_var1211, random_state=randomseed)
    _var1212 = train_y_1.shape
    _var1213 = 1
    N_class = _var1212[_var1213]
    _var1214 = train_x.shape
    _var1215 = 0
    _var1216 = _var1214[_var1215]
    _var1217 = (N_class * N_params)
    _var1218 = (_var1216, _var1217)
    train_blend_x = np.zeros(_var1218)
    _var1219 = test_x.shape
    _var1220 = 0
    _var1221 = _var1219[_var1220]
    _var1222 = (N_class * N_params)
    _var1223 = (_var1221, _var1222)
    test_blend_x = np.zeros(_var1223)
    _var1224 = (fold, N_params)
    scores = np.zeros(_var1224)
    _var1225 = (fold, N_params)
    best_rounds = np.zeros(_var1225)
    fold_start = time.time()
    _var1226 = enumerate(parameters)
    for _var1227 in _var1226:
        _var1230 = 0
        j = _var1227[_var1230]
        _var1231 = 1
        nn_params = _var1227[_var1231]
        _var1232 = test_x.shape
        _var1233 = 0
        _var1234 = _var1232[_var1233]
        _var1235 = (N_class * fold)
        _var1236 = (_var1234, _var1235)
        test_blend_x_j = np.zeros(_var1236)
        _var1237 = skf.split(train_x)
        _var1238 = enumerate(_var1237)
        for _var1239 in _var1238:
            _var1242 = 0
            i = _var1239[_var1242]
            _var1245 = 1
            _var1246 = _var1239[_var1245]
            _var1247 = 0
            train_index = _var1246[_var1247]
            _var1248 = 1
            val_index = _var1246[_var1248]
            train_x_fold = train_x[train_index]
            train_y_fold = train_y_1[train_index]
            val_x_fold = train_x[val_index]
            val_y_fold = train_y_1[val_index]
            model_3 = nn_model(nn_params)
            _var1249 = True
            _var1250 = batch_generator(train_x_fold, train_y_fold, batch_size_1, _var1249)
            _var1251 = 60
            _var1252 = train_x_fold.shape
            _var1253 = 0
            _var1254 = _var1252[_var1253]
            _var1255 = val_x_fold.todense()
            _var1256 = (_var1255, val_y_fold)
            _var1257 = 0
            _var1258 = 'weights.hdf5'
            _var1259 = 'val_loss'
            _var1260 = 0
            _var1261 = True
            _var1262 = ModelCheckpoint(filepath=_var1258, monitor=_var1259, verbose=_var1260, save_best_only=_var1261)
            _var1263 = [_var1262]
            fit_0 = model_3.fit_generator(generator=_var1250, nb_epoch=_var1251, samples_per_epoch=_var1254, validation_data=_var1256, verbose=_var1257, callbacks=_var1263)
            model_4 = fit_0
            _var1264 = fit_0.epoch
            _var1265 = len(_var1264)
            _var1266 = (_var1265 - early_stopping_rounds)
            _var1267 = 1
            best_round = (_var1266 - _var1267)
            _var1268 = (i, j)
            best_rounds_0 = set_index_wrapper(best_rounds, _var1268, best_round)
            _var1269 = 'weights.hdf5'
            model_4.load_weights(_var1269)
            _var1270 = 'categorical_crossentropy'
            _var1271 = 'Adamax'
            model_4.compile(loss=_var1270, optimizer=_var1271)
            _var1272 = val_x_fold.toarray()
            _var1273 = 0
            val_y_predict_fold = model_4.predict_proba(x=_var1272, verbose=_var1273)
            score = log_loss(val_y_fold, val_y_predict_fold)
            _var1274 = (i, j)
            scores_0 = set_index_wrapper(scores, _var1274, score)
            _var1275 = (j * N_class)
            _var1276 = 1
            _var1277 = (j + _var1276)
            _var1278 = (_var1277 * N_class)
            train_blend_x_0 = set_index_wrapper(train_blend_x, (val_index, slice(_var1275, _var1278, None)), val_y_predict_fold)
            _var1279 = 'weights.hdf5'
            model_4.load_weights(_var1279)
            _var1280 = 'categorical_crossentropy'
            _var1281 = 'Adamax'
            model_4.compile(loss=_var1280, optimizer=_var1281)
            _var1282 = (i * N_class)
            _var1283 = 1
            _var1284 = (i + _var1283)
            _var1285 = (_var1284 * N_class)
            _var1286 = test_x.toarray()
            _var1287 = 0
            _var1288 = model_4.predict_proba(x=_var1286, verbose=_var1287)
            test_blend_x_j_0 = set_index_wrapper(test_blend_x_j, (slice(None, None, None), slice(_var1282, _var1285, None)), _var1288)
        test_blend_x_j_1 = __phi__(test_blend_x_j_0, test_blend_x_j)
        best_rounds_1 = __phi__(best_rounds_0, best_rounds)
        fit_1 = __phi__(fit_0, fit)
        model_5 = __phi__(model_4, model_1)
        scores_1 = __phi__(scores_0, scores)
        train_blend_x_1 = __phi__(train_blend_x_0, train_blend_x)
        _var1289 = (j * N_class)
        _var1290 = 1
        _var1291 = (j + _var1290)
        _var1292 = (_var1291 * N_class)
        _var1293 = 0
        _var1294 = (N_class * fold)
        _var1295 = range(_var1293, _var1294, N_class)
        _var1296 = list(_var1295)
        _var1297 = test_blend_x_j_1[:, _var1296]
        _var1298 = 1
        _var1299 = _var1297.mean(_var1298)
        _var1300 = 1
        _var1301 = (N_class * fold)
        _var1302 = range(_var1300, _var1301, N_class)
        _var1303 = list(_var1302)
        _var1304 = test_blend_x_j_1[:, _var1303]
        _var1305 = 1
        _var1306 = _var1304.mean(_var1305)
        _var1307 = 2
        _var1308 = (N_class * fold)
        _var1309 = range(_var1307, _var1308, N_class)
        _var1310 = list(_var1309)
        _var1311 = test_blend_x_j_1[:, _var1310]
        _var1312 = 1
        _var1313 = _var1311.mean(_var1312)
        _var1314 = [_var1299, _var1306, _var1313]
        _var1315 = np.stack(_var1314)
        _var1316 = _var1315.T
        test_blend_x_0 = set_index_wrapper(test_blend_x, (slice(None, None, None), slice(_var1289, _var1292, None)), _var1316)
    best_rounds_2 = __phi__(best_rounds_1, best_rounds)
    fit_2 = __phi__(fit_1, fit)
    train_blend_x_2 = __phi__(train_blend_x_1, train_blend_x)
    model_6 = __phi__(model_5, model_1)
    scores_2 = __phi__(scores_1, scores)
    test_blend_x_1 = __phi__(test_blend_x_0, test_blend_x)
    _var1317 = 'Score for blended models is %f in %0.3fm'
    _var1318 = np.mean(scores_2)
    _var1319 = time.time()
    _var1320 = (_var1319 - fold_start)
    _var1321 = 60
    _var1322 = (_var1320 / _var1321)
    _var1323 = (_var1318, _var1322)
    _var1324 = (_var1317 % _var1323)
    print(_var1324)
    return (train_blend_x_2, test_blend_x_1, scores_2, best_rounds_2)
_var1325 = train_df_nn_0.shape
_var1326 = 0
_var1327 = _var1325[_var1326]
_var1328 = 3
_var1329 = (_var1327, _var1328)
train_total = np.zeros(_var1329)
_var1330 = test_df_nn_0.shape
_var1331 = 0
_var1332 = _var1330[_var1331]
_var1333 = 3
_var1334 = (_var1332, _var1333)
test_total = np.zeros(_var1334)
score_total = 0
count = 100
_var1335 = 'Starting.........'
print(_var1335)
_var1336 = range(count)
for n in _var1336:
    _var1337 = train_X_0.shape
    _var1338 = 1
    _var1339 = _var1337[_var1338]
    _var1340 = {'input_size': 200, 'input_dim': _var1339, 'input_drop_out': 0.4, 'hidden_size': 20, 'hidden_drop_out': 0.4}
    nn_parameters = [_var1340]
    _var1345 = 10
    _var1346 = 5
    _var1347 = 256
    (_var1341, _var1342, _var1343, _var1344) = nn_blend_data(nn_parameters, train_df_nn_0, train_y_0, test_df_nn_0, _var1345, _var1346, _var1347, n)
    train_blend_x_3 = _var1341
    test_blend_x_2 = _var1342
    blend_scores = _var1343
    best_round_0 = _var1344
    train_total_0 = (train_total + train_blend_x_3)
    test_total_0 = (test_total + test_blend_x_2)
    _var1348 = np.mean(blend_scores)
    score_total_0 = (score_total + _var1348)
    name_train_blend = '../tmp/train.csv'
    name_test_blend = '../tmp/test.csv'
    _var1349 = ','
    np.savetxt(name_train_blend, train_total_0, delimiter=_var1349)
    _var1350 = ','
    np.savetxt(name_test_blend, test_total_0, delimiter=_var1350)
score_total_1 = __phi__(score_total_0, score_total)
train_total_1 = __phi__(train_total_0, train_total)
test_total_1 = __phi__(test_total_0, test_total)
train_total_2 = (train_total_1 / count)
test_total_2 = (test_total_1 / count)
score_total_2 = (score_total_1 / count)
test_total_2
now_0 = datetime.now()
_var1351 = '../output/sub_2ndKeras_100bagging_'
_var1352 = '%Y-%m-%d-%H-%M'
_var1353 = now_0.strftime(_var1352)
_var1354 = str(_var1353)
_var1355 = (_var1351 + _var1354)
_var1356 = '.csv'
sub_name_0 = (_var1355 + _var1356)
out_df_2 = pd.DataFrame(test_total_2)
_var1357 = ['low', 'medium', 'high']
out_df_3 = set_field_wrapper(out_df_2, 'columns', _var1357)
_var1358 = 'listing_id'
out_df_4 = set_index_wrapper(out_df_3, _var1358, sub_id)
_var1359 = False
out_df_4.to_csv(sub_name_0, index=_var1359)
_var1360 = '../output/train_blend_2ndKeras_100bagging_'
_var1361 = '%Y-%m-%d-%H-%M'
_var1362 = now_0.strftime(_var1361)
_var1363 = str(_var1362)
_var1364 = (_var1360 + _var1363)
_var1365 = '.csv'
name_train_blend_0 = (_var1364 + _var1365)
_var1366 = '../output/test_blend_2ndKeras_100bagging_'
_var1367 = '%Y-%m-%d-%H-%M'
_var1368 = now_0.strftime(_var1367)
_var1369 = str(_var1368)
_var1370 = (_var1366 + _var1369)
_var1371 = '.csv'
name_test_blend_0 = (_var1370 + _var1371)
_var1372 = 0
_var1373 = np.mean(blend_scores, axis=_var1372)
print(_var1373)
_var1374 = 0
_var1375 = np.mean(best_round_0, axis=_var1374)
print(_var1375)
_var1376 = ','
np.savetxt(name_train_blend_0, train_total_2, delimiter=_var1376)
_var1377 = ','
np.savetxt(name_test_blend_0, test_total_2, delimiter=_var1377)
