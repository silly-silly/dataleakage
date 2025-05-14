

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
from datetime import date
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re
from sklearn.feature_selection import *
from sklearn.linear_model import *
from sklearn.neighbors import *
from sklearn.svm import *
from sklearn.neighbors import *
from sklearn.tree import *
from sklearn.preprocessing import *
from xgboost import *
from sklearn.metrics import *
from geopy.distance import great_circle
import json
from pandas.io.json import json_normalize
from statsmodels.tsa.seasonal import seasonal_decompose
import requests
import descartes
import math
_var0 = 'Libraries imported.'
print(_var0)
from google.colab import drive
_var1 = '/content/gdrive'
drive.mount(_var1)
_var2 = '/content/gdrive/My Drive/listingss.csv'
df = pd.read_csv(_var2)
_var3 = len(df)
_var4 = f'the numer of observations are {_var3}'
print(_var4)
categoricals = [var for var in df.columns if (df[var].dtype == 'object')]
numerics = [var for var in df.columns if ((df[var].dtype == 'int64') | (df[var].dtype == 'float64'))]
dates = [var for var in df.columns if (df[var].dtype == 'datetime64[ns]')]
one_hot_col_names = ['host_id', 'host_location', 'host_response_time', 'host_is_superhost', 'host_neighbourhood', 'host_has_profile_pic', 'host_identity_verified', 'neighbourhood', 'neighbourhood_cleansed', 'neighbourhood_group_cleansed', 'zipcode', 'is_location_exact', 'property_type', 'room_type', 'bed_type', 'has_availability', 'requires_license', 'instant_bookable', 'is_business_travel_ready', 'cancellation_policy', 'cancellation_policy', 'require_guest_profile_picture', 'require_guest_phone_verification', 'calendar_updated']
text_cols = ['name', 'summary', 'space', 'description', 'neighborhood_overview', 'notes', 'transit', 'access', 'interaction', 'house_rules', 'host_name', 'host_about']
features = ['host_listings_count', 'host_total_listings_count', 'latitude', 'longitude', 'accommodates', 'bathrooms', 'bedrooms', 'beds', 'square_feet', 'guests_included', 'minimum_nights', 'maximum_nights', 'availability_30', 'availability_60', 'availability_90', 'availability_365', 'number_of_reviews', 'review_scores_rating', 'review_scores_accuracy', 'review_scores_cleanliness', 'review_scores_checkin', 'review_scores_communication', 'review_scores_location', 'review_scores_value', 'calculated_host_listings_count', 'reviews_per_month']
price_features = ['security_deposit', 'cleaning_fee', 'extra_people', 'price']
date_cols = ['host_since', 'first_review', 'last_review']

def host_verification(cols):
    possible_words = {}
    i = 0
    for col in cols:
        words = col.split()
        for w in words:
            _var5 = '\\W+'
            _var6 = ''
            wr = re.sub(_var5, _var6, w)
            _var7 = ''
            _var8 = (wr != _var7)
            _var9 = (wr not in possible_words)
            _var10 = (_var8 and _var9)
            if _var10:
                possible_words_0 = set_index_wrapper(possible_words, wr, i)
                _var11 = 1
                i_0 = (i + _var11)
            possible_words_1 = __phi__(possible_words_0, possible_words)
            i_1 = __phi__(i_0, i)
        possible_words_2 = __phi__(possible_words_1, possible_words)
        i_2 = __phi__(i_1, i)
    possible_words_3 = __phi__(possible_words_2, possible_words)
    i_3 = __phi__(i_2, i)
    l = len(possible_words_3)
    _var12 = cols.shape
    _var13 = 0
    _var14 = _var12[_var13]
    _var15 = (_var14, l)
    new_cols = np.zeros(_var15)
    _var16 = enumerate(cols)
    for _var17 in _var16:
        _var20 = 0
        i_4 = _var17[_var20]
        _var21 = 1
        col_0 = _var17[_var21]
        words_0 = col_0.split()
        arr = np.zeros(l)
        for w_0 in words_0:
            _var22 = '\\W+'
            _var23 = ''
            wr_0 = re.sub(_var22, _var23, w_0)
            _var24 = ''
            _var25 = (wr_0 != _var24)
            if _var25:
                _var26 = possible_words_3[wr_0]
                _var27 = 1
                arr_0 = set_index_wrapper(arr, _var26, _var27)
            arr_1 = __phi__(arr_0, arr)
        arr_2 = __phi__(arr_1, arr)
        wr_1 = __phi__(wr_0, wr)
        new_cols_0 = set_index_wrapper(new_cols, i_4, arr_2)
    words_1 = __phi__(words_0, words)
    col_1 = __phi__(col_0, col)
    w_1 = __phi__(w_0, w)
    new_cols_1 = __phi__(new_cols_0, new_cols)
    i_5 = __phi__(i_4, i_3)
    wr_2 = __phi__(wr_1, wr)
    return new_cols_1

def amenities(cols_0):
    dic = {}
    i_6 = 0
    for col_2 in cols_0:
        _var28 = ','
        arr_3 = col_2.split(_var28)
        for a in arr_3:
            _var29 = '\\W+'
            _var30 = ''
            ar = re.sub(_var29, _var30, a)
            _var31 = len(ar)
            _var32 = 0
            _var33 = (_var31 > _var32)
            if _var33:
                _var34 = (ar not in dic)
                if _var34:
                    dic_0 = set_index_wrapper(dic, ar, i_6)
                    _var35 = 1
                    i_7 = (i_6 + _var35)
                dic_1 = __phi__(dic_0, dic)
                i_8 = __phi__(i_7, i_6)
            dic_2 = __phi__(dic_1, dic)
            i_9 = __phi__(i_8, i_6)
        dic_3 = __phi__(dic_2, dic)
        i_10 = __phi__(i_9, i_6)
    dic_4 = __phi__(dic_3, dic)
    i_11 = __phi__(i_10, i_6)
    l_0 = len(dic_4)
    _var36 = cols_0.shape
    _var37 = 0
    _var38 = _var36[_var37]
    _var39 = (_var38, l_0)
    new_cols_2 = np.zeros(_var39)
    _var40 = enumerate(cols_0)
    for _var41 in _var40:
        _var44 = 0
        i_12 = _var41[_var44]
        _var45 = 1
        col_3 = _var41[_var45]
        _var46 = ','
        words_2 = col_3.split(_var46)
        arr_4 = np.zeros(l_0)
        for w_2 in words_2:
            _var47 = '\\W+'
            _var48 = ''
            wr_3 = re.sub(_var47, _var48, w_2)
            _var49 = ''
            _var50 = (wr_3 != _var49)
            if _var50:
                _var51 = dic_4[wr_3]
                _var52 = 1
                arr_5 = set_index_wrapper(arr_4, _var51, _var52)
            arr_6 = __phi__(arr_5, arr_4)
        arr_7 = __phi__(arr_6, arr_4)
        new_cols_3 = set_index_wrapper(new_cols_2, i_12, arr_7)
    arr_8 = __phi__(arr_7, arr_3)
    i_13 = __phi__(i_12, i_11)
    new_cols_4 = __phi__(new_cols_3, new_cols_2)
    col_4 = __phi__(col_3, col_2)
    return new_cols_4

def one_hot(arr_9):
    _var53 = global_wrapper(LabelEncoder)
    label_encoder = _var53()
    integer_encoded = label_encoder.fit_transform(arr_9)
    _var54 = global_wrapper(OneHotEncoder)
    _var55 = False
    onehot_encoder = _var54(sparse=_var55)
    _var56 = len(integer_encoded)
    _var57 = 1
    integer_encoded_0 = integer_encoded.reshape(_var56, _var57)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded_0)
    return onehot_encoded
one_hot_col_names_0 = ['host_response_time', 'host_is_superhost', 'host_has_profile_pic', 'host_identity_verified', 'neighbourhood_cleansed', 'neighbourhood_group_cleansed', 'zipcode', 'is_location_exact', 'property_type', 'room_type', 'bed_type', 'has_availability', 'requires_license', 'instant_bookable', 'is_business_travel_ready', 'cancellation_policy', 'require_guest_profile_picture', 'require_guest_phone_verification', 'calendar_updated']
one_hot_dict = {}
for i_14 in one_hot_col_names_0:
    _var58 = df[i_14]
    _var59 = ''
    _var60 = _var58.fillna(_var59)
    _var61 = np.array(_var60, dtype=str)
    _var62 = one_hot(_var61)
    one_hot_dict_0 = set_index_wrapper(one_hot_dict, i_14, _var62)
one_hot_dict_1 = __phi__(one_hot_dict_0, one_hot_dict)
_var63 = 'host_verifications'
_var64 = 'host_verifications'
_var65 = df[_var64]
_var66 = host_verification(_var65)
one_hot_dict_2 = set_index_wrapper(one_hot_dict_1, _var63, _var66)
_var67 = 'amenities'
_var68 = 'amenities'
_var69 = df[_var68]
_var70 = amenities(_var69)
one_hot_dict_3 = set_index_wrapper(one_hot_dict_2, _var67, _var70)
ont_hot_list = []
_var71 = one_hot_dict_3.keys()
_var72 = list(_var71)
for i_15 in _var72:
    _var73 = 1
    _var74 = one_hot_dict_3[i_15]
    _var75 = _var74.shape
    _var76 = 1
    _var77 = _var75[_var76]
    _var78 = 400
    _var79 = (_var73 < _var77 < _var78)
    if _var79:
        _var80 = one_hot_dict_3[i_15]
        ont_hot_list.append(_var80)
_var81 = 1
onehot_variables = np.concatenate(ont_hot_list, axis=_var81)
hot_cat_variables = pd.DataFrame(onehot_variables)
_var82 = hot_cat_variables.isnull()
_var83 = _var82.sum()
_var83.sum()
hot_cat_variables.shape

def check_nan(cols_1):
    for col_5 in cols_1:
        _var84 = np.isnan(col_5)
        if _var84:
            _var85 = True
            return _var85
    _var86 = False
    return _var86

def clean_host_response_rate(host_response_rate, num_data):
    total = 0
    count = 0
    for col_6 in host_response_rate:
        _var87 = isinstance(col_6, float)
        _var88 = (not _var87)
        if _var88:
            _var89 = '%'
            _var90 = col_6.strip(_var89)
            _var91 = float(_var90)
            total_0 = (total + _var91)
            _var92 = 1
            count_0 = (count + _var92)
        total_1 = __phi__(total_0, total)
        count_1 = __phi__(count_0, count)
    total_2 = __phi__(total_1, total)
    count_2 = __phi__(count_1, count)
    arr_10 = np.zeros(num_data)
    mean = (total_2 / count_2)
    _var93 = enumerate(host_response_rate)
    for _var94 in _var93:
        _var97 = 0
        i_16 = _var94[_var97]
        _var98 = 1
        col_7 = _var94[_var98]
        _var99 = isinstance(col_7, float)
        _var100 = (not _var99)
        if _var100:
            _var101 = arr_10[i_16]
            _var102 = '%'
            _var103 = col_7.strip(_var102)
            _var104 = float(_var103)
            _var105 = (_var101 + _var104)
            arr_11 = set_index_wrapper(arr_10, i_16, _var105)
        else:
            _var106 = math.isnan(col_7)
            assert _var106
            arr_12 = set_index_wrapper(arr_10, i_16, mean)
        arr_13 = __phi__(arr_11, arr_12)
    arr_14 = __phi__(arr_13, arr_10)
    i_17 = __phi__(i_16, i_15)
    col_8 = __phi__(col_7, col_6)
    return arr_14

def clean_price(price, num_data_0):
    arr_15 = np.zeros(num_data_0)
    _var107 = enumerate(price)
    for _var108 in _var107:
        _var111 = 0
        i_18 = _var108[_var111]
        _var112 = 1
        col_9 = _var108[_var112]
        _var113 = isinstance(col_9, float)
        _var114 = (not _var113)
        if _var114:
            _var115 = arr_15[i_18]
            _var116 = '$'
            _var117 = col_9.strip(_var116)
            _var118 = ','
            _var119 = ''
            _var120 = _var117.replace(_var118, _var119)
            _var121 = float(_var120)
            _var122 = (_var115 + _var121)
            arr_16 = set_index_wrapper(arr_15, i_18, _var122)
        else:
            _var123 = math.isnan(col_9)
            assert _var123
            _var124 = 0
            arr_17 = set_index_wrapper(arr_15, i_18, _var124)
        arr_18 = __phi__(arr_16, arr_17)
    arr_19 = __phi__(arr_18, arr_15)
    i_19 = __phi__(i_18, i_15)
    return arr_19

def to_np_array_fill_NA_mean(cols_2):
    _var125 = np.array(cols_2)
    _var126 = np.nanmean(_var125)
    _var127 = cols_2.fillna(_var126)
    _var128 = np.array(_var127)
    return _var128
_var129 = df.shape
_var130 = 0
num_data_1 = _var129[_var130]
_var131 = len(features)
_var132 = len(price_features)
_var133 = (_var131 + _var132)
_var134 = 1
_var135 = (_var133 + _var134)
_var136 = (_var135, num_data_1)
arr_20 = np.zeros(_var136)
_var137 = 'host_response_rate'
_var138 = df[_var137]
host_response_rate_0 = clean_host_response_rate(_var138, num_data_1)
_var139 = 0
arr_21 = set_index_wrapper(arr_20, _var139, host_response_rate_0)
i_20 = 0
for feature in features:
    _var140 = 1
    i_21 = (i_20 + _var140)
    _var141 = df[feature]
    _var142 = check_nan(_var141)
    if _var142:
        _var143 = df[feature]
        _var144 = to_np_array_fill_NA_mean(_var143)
        arr_22 = set_index_wrapper(arr_21, i_21, _var144)
    else:
        _var145 = df[feature]
        _var146 = np.array(_var145)
        arr_23 = set_index_wrapper(arr_21, i_21, _var146)
    arr_24 = __phi__(arr_22, arr_23)
arr_25 = __phi__(arr_24, arr_21)
i_22 = __phi__(i_21, i_20)
for feature_0 in price_features:
    _var147 = 1
    i_23 = (i_22 + _var147)
    _var148 = df[feature_0]
    _var149 = clean_price(_var148, num_data_1)
    arr_26 = set_index_wrapper(arr_25, i_23, _var149)
arr_27 = __phi__(arr_26, arr_25)
i_24 = __phi__(i_23, i_22)
_var150 = (- 1)
target = arr_27[_var150]
_var151 = (- 1)
_var152 = arr_27[:_var151]
numeric_variables = _var152.T
numeric_variables_0 = pd.DataFrame(numeric_variables)
_var153 = numeric_variables_0.isnull()
_var154 = _var153.sum()
_var154.sum()
_var155 = (numeric_variables_0, hot_cat_variables)
_var156 = 1
inde_variables = np.concatenate(_var155, axis=_var156)
inde_variables_0 = pd.DataFrame(inde_variables)
_var157 = inde_variables_0.isnull()
_var158 = _var157.sum()
_var158.sum()
_var159 = 0
mean_0 = np.mean(inde_variables_0, axis=_var159)
_var160 = 0
std = np.std(inde_variables_0, axis=_var160)
_var161 = (inde_variables_0 - mean_0)
inde_variables_1 = (_var161 / std)
inde_variables_1.shape
import torch
from torch import nn
import torch.optim as optim
import numpy as np
import random
import copy
import torch.utils.data as data
import os
_var162 = nn.Module

class NN229(_var162):

    def __init__(self, input_size=355, hidden_size1=128, hidden_size2=512, hidden_size3=64, output_size=1, drop_prob=0.05):
        _var163 = super(NN229, self)
        _var163.__init__()
        _var164 = nn.ReLU()
        self_0 = set_field_wrapper(self, 'relu', _var164)
        _var165 = nn.Dropout(p=drop_prob)
        self_1 = set_field_wrapper(self_0, 'dropout', _var165)
        _var166 = nn.Linear(input_size, hidden_size1)
        self_2 = set_field_wrapper(self_1, 'W1', _var166)
        _var167 = nn.Linear(hidden_size1, hidden_size2)
        self_3 = set_field_wrapper(self_2, 'W2', _var167)
        _var168 = nn.Linear(hidden_size2, hidden_size3)
        self_4 = set_field_wrapper(self_3, 'W3', _var168)
        _var169 = nn.Linear(hidden_size3, output_size)
        self_5 = set_field_wrapper(self_4, 'W4', _var169)

    def forward(self_6, x):
        _var170 = self_6.W1(x)
        _var171 = self_6.relu(_var170)
        hidden1 = self_6.dropout(_var171)
        _var172 = self_6.W2(hidden1)
        _var173 = self_6.relu(_var172)
        hidden2 = self_6.dropout(_var173)
        _var174 = self_6.W3(hidden2)
        _var175 = self_6.relu(_var174)
        hidden3 = self_6.dropout(_var175)
        out = self_6.W4(hidden3)
        return out
_var176 = data.Dataset

class AirBnb(_var176):

    def __init__(self_7, train_path, label_path):
        _var177 = super(AirBnb, self_7)
        _var177.__init__()
        _var178 = torch.from_numpy(train_path)
        _var179 = _var178.float()
        self_8 = set_field_wrapper(self_7, 'x', _var179)
        _var180 = torch.from_numpy(label_path)
        _var181 = _var180.float()
        self_9 = set_field_wrapper(self_8, 'y', _var181)

    def __getitem__(self_10, idx):
        _var182 = self_10.x
        x_0 = _var182[idx]
        _var183 = self_10.y
        y = _var183[idx]
        return (x_0, y)

    def __len__(self_11):
        _var184 = self_11.x
        _var185 = _var184.shape
        _var186 = 0
        _var187 = _var185[_var186]
        return _var187
_var188 = data.Dataset

class CSVDataset(_var188):

    def __init__(self_12, train_path_0, label_path_0):
        _var189 = super(CSVDataset, self_12)
        _var189.__init__()
        _var190 = torch.from_numpy(train_path_0)
        _var191 = _var190.float()
        self_13 = set_field_wrapper(self_12, 'x', _var191)
        _var192 = torch.from_numpy(label_path_0)
        _var193 = _var192.float()
        self_14 = set_field_wrapper(self_13, 'y', _var193)
        _var194 = self_14.y
        _var195 = self_14.y
        _var196 = len(_var195)
        _var197 = 1
        _var198 = (_var196, _var197)
        _var199 = _var194.reshape(_var198)
        self_15 = set_field_wrapper(self_14, 'y', _var199)

    def __len__(self_16):
        _var200 = self_16.x
        _var201 = len(_var200)
        return _var201

    def __getitem__(self_17, idx_0):
        _var202 = self_17.x
        _var203 = _var202[idx_0]
        _var204 = self_17.y
        _var205 = _var204[idx_0]
        _var206 = [_var203, _var205]
        return _var206

    def get_splits(self_18, n_test=0.33):
        _var207 = self_18.x
        _var208 = len(_var207)
        _var209 = (n_test * _var208)
        test_size = round(_var209)
        _var210 = self_18.x
        _var211 = len(_var210)
        train_size = (_var211 - test_size)
        _var212 = [train_size, test_size]
        _var213 = data.random_split(self_18, _var212)
        return _var213

def load_model(model, optimizer, checkpoint_path, model_only=False):
    _var214 = 'cuda:0'
    ckpt_dict = torch.load(checkpoint_path, map_location=_var214)
    _var215 = 'state_dict'
    _var216 = ckpt_dict[_var215]
    model.load_state_dict(_var216)
    _var217 = (not model_only)
    if _var217:
        _var218 = 'optimizer'
        _var219 = ckpt_dict[_var218]
        optimizer.load_state_dict(_var219)
        _var220 = 'epoch'
        epoch = ckpt_dict[_var220]
        _var221 = 'val_loss'
        val_loss = ckpt_dict[_var221]
    else:
        epoch_0 = None
        val_loss_0 = None
    val_loss_1 = __phi__(val_loss, val_loss_0)
    epoch_1 = __phi__(epoch, epoch_0)
    return (model, optimizer, epoch_1, val_loss_1)
np.log(target)

def train(model_0: NN229, optimizer_0, loss_fn, epoch_2=0):
    _var222 = global_wrapper(inde_variables_1)
    _var223 = _var222.to_numpy()
    _var224 = global_wrapper(target)
    train_dataset = CSVDataset(_var223, _var224)
    (_var225, _var226) = train_dataset.get_splits()
    train = _var225
    test = _var226
    _var227 = global_wrapper(batch_size)
    _var228 = True
    train_loader = data.DataLoader(train, batch_size=_var227, shuffle=_var228)
    _var229 = global_wrapper(batch_size)
    _var230 = True
    dev_loader = data.DataLoader(test, batch_size=_var229, shuffle=_var230)
    model_0.train()
    step = 0
    best_model = NN229()
    best_epoch = 0
    best_val_loss = None
    _var231 = global_wrapper(max_epoch)
    _var232 = (epoch_2 < _var231)
    while _var232:
        _var233 = 1
        epoch_3 = (epoch_2 + _var233)
        stats = []
        with torch.enable_grad():
            for _var234 in train_loader:
                _var237 = 0
                x_1 = _var234[_var237]
                _var238 = 1
                y_0 = _var234[_var238]
                _var239 = 1
                step_0 = (step + _var239)
                x_2 = x_1.cuda()
                y_1 = y_0.cuda()
                optimizer_0.zero_grad()
                _var240 = model_0(x_2)
                _var241 = (- 1)
                pred = _var240.reshape(_var241)
                loss = loss_fn(pred, y_1)
                loss_val = loss.item()
                loss.backward()
                optimizer_0.step()
                stats.append(loss_val)
            step_1 = __phi__(step_0, step)
        step_2 = __phi__(step_1, step)
        _var242 = 'Train loss: '
        _var243 = sum(stats)
        _var244 = len(stats)
        _var245 = (_var243 / _var244)
        _var246 = (_var242, _var245)
        print(_var246)
        _var247 = global_wrapper(evaluate)
        val_loss_2 = _var247(dev_loader, model_0)
        _var248 = None
        _var249 = (best_val_loss is _var248)
        _var250 = (best_val_loss > val_loss_2)
        _var251 = (_var249 or _var250)
        if _var251:
            best_val_loss_0 = val_loss_2
            model_0.cpu()
            best_model_0 = copy.deepcopy(model_0)
            model_0.cuda()
            best_epoch_0 = epoch_3
        best_epoch_1 = __phi__(best_epoch_0, best_epoch)
        best_val_loss_1 = __phi__(best_val_loss_0, best_val_loss)
        best_model_1 = __phi__(best_model_0, best_model)
    best_epoch_2 = __phi__(best_epoch_1, best_epoch)
    step_3 = __phi__(step_2, step)
    best_val_loss_2 = __phi__(best_val_loss_1, best_val_loss)
    epoch_4 = __phi__(epoch_3, epoch_2)
    best_model_2 = __phi__(best_model_1, best_model)
    return (best_model_2, best_epoch_2, best_val_loss_2)

def evaluate(dev_loader_0, model_1: NN229):
    model_1.eval()
    stats_0 = []
    with torch.no_grad():
        for _var252 in dev_loader_0:
            _var255 = 0
            x_3 = _var252[_var255]
            _var256 = 1
            y_2 = _var252[_var256]
            x_4 = x_3.cuda()
            y_3 = y_2.cuda()
            _var257 = model_1(x_4)
            _var258 = (- 1)
            pred_0 = _var257.reshape(_var258)
            _var259 = global_wrapper(loss_fn)
            _var260 = _var259(pred_0, y_3)
            loss_val_0 = _var260.item()
            stats_0.append(loss_val_0)
    _var261 = 'Val loss: '
    _var262 = sum(stats_0)
    _var263 = len(stats_0)
    _var264 = (_var262 / _var263)
    _var265 = (_var261, _var264)
    print(_var265)
    _var266 = sum(stats_0)
    _var267 = len(stats_0)
    _var268 = (_var266 / _var267)
    return _var268
lr = 0.0001
weight_decay = 1e-05
_var269 = 0.9
_var270 = 0.999
beta = (_var269, _var270)
max_epoch_0 = 100
batch_size_0 = 64
_var271 = NN229()
model_2 = _var271.cuda()
_var272 = model_2.parameters()
optimizer_1 = optim.Adam(_var272, lr=lr, weight_decay=weight_decay, betas=beta)
loss_fn_0 = nn.MSELoss()
_var276 = 0
(_var273, _var274, _var275) = train(model_2, optimizer_1, loss_fn_0, epoch_2=_var276)
best_model_3 = _var273
best_epoch_3 = _var274
best_val_loss_3 = _var275
_var277 = inde_variables_1.to_numpy()
train_dataset_0 = CSVDataset(_var277, target)
(_var278, _var279) = train_dataset_0.get_splits()
train_0 = _var278
test_0 = _var279
_var280 = True
dev_loader_1 = data.DataLoader(test_0, shuffle=_var280)
y_truth_list = []
for _var281 in dev_loader_1:
    _var284 = 0
    _ = _var281[_var284]
    _var285 = 1
    y_truth = _var281[_var285]
    _var286 = 0
    _var287 = y_truth[_var286]
    _var288 = 0
    _var289 = _var287[_var288]
    _var290 = _var289.cpu()
    _var291 = _var290.numpy()
    y_truth_list.append(_var291)
y_pred_list = [a.squeeze().tolist() for a in y_truth_list]
y_t = np.array(y_truth_list)
y_t
_var292 = torch.cuda
_var293 = _var292.is_available()
_var294 = 'cuda:0'
_var295 = 'cpu'
_var296 = (_var294 if _var293 else _var295)
device = torch.device(_var296)
y_pred_list_0 = []
with torch.no_grad():
    model_2.eval()
    for _var297 in dev_loader_1:
        _var300 = 0
        X_batch = _var297[_var300]
        _var301 = 1
        __0 = _var297[_var301]
        X_batch_0 = X_batch.to(device)
        y_test_pred = model_2(X_batch_0)
        _var302 = y_test_pred.cpu()
        _var303 = _var302.numpy()
        y_pred_list_0.append(_var303)
    __1 = __phi__(__0, _)
__2 = __phi__(__1, _)
y_pred_list_1 = [a.squeeze().tolist() for a in y_pred_list]
y_p = np.array(y_pred_list_1)
y_p
import sklearn.metrics
_var304 = global_wrapper(sklearn)
_var305 = _var304.metrics
_var305.r2_score(y_t, y_p)
