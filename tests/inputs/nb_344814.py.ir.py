

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
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
_var0 = get_ipython()
_var1 = 'matplotlib'
_var2 = 'inline'
_var0.run_line_magic(_var1, _var2)
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
import patsy
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import GridSearchCV
_var3 = '/Users/timothyernst/Documents/General Assembly/rewebscraping/scrapeSan Francisco_1000.csv'
San_Fran = pd.read_csv(_var3)
_var4 = '/Users/timothyernst/Documents/General Assembly/rewebscraping/scrapeAtlanta_1000.csv'
Atlanta = pd.read_csv(_var4)
_var5 = '/Users/timothyernst/Documents/General Assembly/rewebscraping/scrapeAustin_1000.csv'
Austin = pd.read_csv(_var5)
_var6 = '/Users/timothyernst/Documents/General Assembly/rewebscraping/scrapeBoston_1000.csv'
Boston = pd.read_csv(_var6)
_var7 = '/Users/timothyernst/Documents/General Assembly/rewebscraping/scrapeChicago_1000.csv'
Chicago = pd.read_csv(_var7)
_var8 = '/Users/timothyernst/Documents/General Assembly/rewebscraping/scrapeNew York_1000.csv'
New_York = pd.read_csv(_var8)
_var9 = '/Users/timothyernst/Documents/General Assembly/rewebscraping/scrapeSeattle_1000.csv'
Seattle = pd.read_csv(_var9)
_var10 = '/Users/timothyernst/Documents/General Assembly/newcities/scrapeDenver_1000.csv'
Denver = pd.read_csv(_var10)
_var11 = '/Users/timothyernst/Documents/General Assembly/newcities/scrapeHouston_1000.csv'
Houston = pd.read_csv(_var11)
_var12 = '/Users/timothyernst/Documents/General Assembly/newcities/scrapeLos_Angeles_1000.csv'
Los_Angeles = pd.read_csv(_var12)
_var13 = '/Users/timothyernst/Documents/General Assembly/newcities/scrapePortland_OR_1000.csv'
Portland = pd.read_csv(_var13)
_var14 = '/Users/timothyernst/Documents/General Assembly/newcities/scrapeRaleigh_1000.csv'
Raleigh = pd.read_csv(_var14)
_var15 = '/Users/timothyernst/Documents/General Assembly/newcities/scrapeSalt_Lake_City_1000.csv'
Salt_Lake_City = pd.read_csv(_var15)
_var16 = '/Users/timothyernst/Documents/General Assembly/newcities/scrapeWashington_DC_1000.csv'
Washington_DC = pd.read_csv(_var16)
_var17 = [San_Fran, Atlanta, Austin, Boston, Chicago, New_York, Seattle, Denver, Houston, Los_Angeles, Portland, Raleigh, Salt_Lake_City, Washington_DC]
city_data = pd.concat(_var17)
city_data.info()
_var18 = ['location', 'company', 'title', 'salary']
df = city_data.dropna(subset=_var18)
_var19 = ['jkid']
df_0 = df.drop_duplicates(subset=_var19)
df_0.info()
_var20 = 'min_sal'
_var21 = [x.split(' ')[0] for x in df.salary]
df_1 = set_index_wrapper(df_0, _var20, _var21)
_var22 = 'max_sal'
_var23 = [x.split(' ')[(- 3)] for x in df.salary]
df_2 = set_index_wrapper(df_1, _var22, _var23)
_var24 = 'unit'
_var25 = [x.split(' ')[(- 1)] for x in df.salary]
df_3 = set_index_wrapper(df_2, _var24, _var25)
_var26 = 'min_sal'
_var27 = [x.replace('$', '') for x in df.min_sal]
df_4 = set_index_wrapper(df_3, _var26, _var27)
_var28 = 'max_sal'
_var29 = [x.replace('$', '') for x in df.max_sal]
df_5 = set_index_wrapper(df_4, _var28, _var29)
_var30 = 'min_sal'
_var31 = [float(x.replace(',', '')) for x in df.min_sal]
df_6 = set_index_wrapper(df_5, _var30, _var31)
_var32 = 'max_sal'
_var33 = [float(x.replace(',', '')) for x in df.max_sal]
df_7 = set_index_wrapper(df_6, _var32, _var33)
df_7.info()
_var34 = 'max_sal'
_var35 = 'max_sal'
_var36 = df_7[_var35]
_var37 = _var36.astype(float)
df_8 = set_index_wrapper(df_7, _var34, _var37)
_var38 = 'avg_sal'
_var39 = 'min_sal'
_var40 = df_8[_var39]
_var41 = 'max_sal'
_var42 = df_8[_var41]
_var43 = (_var40 + _var42)
_var44 = 2
_var45 = (_var43 / _var44)
df_9 = set_index_wrapper(df_8, _var38, _var45)
dict_unit = {'month': 12, 'hour': 2000, 'year': 1, 'week': 50, 'day': 250}
_var46 = 'm'
_var47 = 'unit'
_var48 = df_9[_var47]

def _func0(x):
    _var49 = global_wrapper(dict_unit)
    _var50 = _var49[x]
    return _var50
_var51 = _var48.map(_func0)
df_10 = set_index_wrapper(df_9, _var46, _var51)
_var52 = 'ann_sal'
_var53 = 'avg_sal'
_var54 = df_10[_var53]
_var55 = 'm'
_var56 = df_10[_var55]
_var57 = (_var54 * _var56)
df_11 = set_index_wrapper(df_10, _var52, _var57)
_var58 = 'state'
_var59 = [x.split(', ')[1] for x in df['location']]
df_12 = set_index_wrapper(df_11, _var58, _var59)
_var60 = 'city'
_var61 = [x.split(', ')[0] for x in df['location']]
df_13 = set_index_wrapper(df_12, _var60, _var61)
_var62 = 'state'
_var63 = [x.split(' ')[0] for x in df['state']]
df_14 = set_index_wrapper(df_13, _var62, _var63)

def title_category(words):
    _var64 = 'scientist'
    _var65 = words.lower()
    _var66 = (_var64 in _var65)
    _var67 = 'data'
    _var68 = words.lower()
    _var69 = (_var67 in _var68)
    _var70 = 'senior'
    _var71 = words.lower()
    _var72 = (_var70 in _var71)
    _var73 = (_var66 and _var69 and _var72)
    if _var73:
        _var74 = 'senior data scientist'
        return _var74
    else:
        _var75 = 'science'
        _var76 = words.lower()
        _var77 = (_var75 in _var76)
        _var78 = 'director'
        _var79 = words.lower()
        _var80 = (_var78 in _var79)
        _var81 = 'data'
        _var82 = words.lower()
        _var83 = (_var81 in _var82)
        _var84 = (_var77 and _var80 and _var83)
        if _var84:
            _var85 = 'senior data scientist'
            return _var85
        else:
            _var86 = 'scientist'
            _var87 = words.lower()
            _var88 = (_var86 in _var87)
            _var89 = 'lead'
            _var90 = words.lower()
            _var91 = (_var89 in _var90)
            _var92 = (_var88 and _var91)
            if _var92:
                _var93 = 'senior data scientist'
                return _var93
            else:
                _var94 = 'scientist'
                _var95 = words.lower()
                _var96 = (_var94 in _var95)
                _var97 = 'sr'
                _var98 = words.lower()
                _var99 = (_var97 in _var98)
                _var100 = (_var96 and _var99)
                if _var100:
                    _var101 = 'senior data scientist'
                    return _var101
                else:
                    _var102 = 'science'
                    _var103 = words.lower()
                    _var104 = (_var102 in _var103)
                    _var105 = 'manager'
                    _var106 = words.lower()
                    _var107 = (_var105 in _var106)
                    _var108 = 'data'
                    _var109 = words.lower()
                    _var110 = (_var108 in _var109)
                    _var111 = (_var104 and _var107 and _var110)
                    if _var111:
                        _var112 = 'senior data scientist'
                        return _var112
                    else:
                        _var113 = 'scientist'
                        _var114 = words.lower()
                        _var115 = (_var113 in _var114)
                        _var116 = 'data'
                        _var117 = words.lower()
                        _var118 = (_var116 in _var117)
                        _var119 = (_var115 and _var118)
                        if _var119:
                            _var120 = 'data scientist'
                            return _var120
                        else:
                            _var121 = 'learning'
                            _var122 = words.lower()
                            _var123 = (_var121 in _var122)
                            _var124 = 'machine'
                            _var125 = words.lower()
                            _var126 = (_var124 in _var125)
                            _var127 = 'scientist'
                            _var128 = words.lower()
                            _var129 = (_var127 in _var128)
                            _var130 = (_var123 and _var126 and _var129)
                            if _var130:
                                _var131 = 'data scientist'
                                return _var131
                            else:
                                _var132 = 'analyst'
                                _var133 = words.lower()
                                _var134 = (_var132 in _var133)
                                _var135 = 'data'
                                _var136 = words.lower()
                                _var137 = (_var135 in _var136)
                                _var138 = (_var134 and _var137)
                                if _var138:
                                    _var139 = 'data analyst'
                                    return _var139
                                else:
                                    _var140 = 'analyst'
                                    _var141 = words.lower()
                                    _var142 = (_var140 in _var141)
                                    _var143 = 'quantitative'
                                    _var144 = words.lower()
                                    _var145 = (_var143 in _var144)
                                    _var146 = (_var142 and _var145)
                                    if _var146:
                                        _var147 = 'data analyst'
                                        return _var147
                                    else:
                                        _var148 = 'analytics'
                                        _var149 = words.lower()
                                        _var150 = (_var148 in _var149)
                                        _var151 = 'manager'
                                        _var152 = words.lower()
                                        _var153 = (_var151 in _var152)
                                        _var154 = (_var150 and _var153)
                                        if _var154:
                                            _var155 = 'data analyst'
                                            return _var155
                                        else:
                                            _var156 = 'analyst'
                                            _var157 = words.lower()
                                            _var158 = (_var156 in _var157)
                                            _var159 = 'research'
                                            _var160 = words.lower()
                                            _var161 = (_var159 in _var160)
                                            _var162 = (_var158 and _var161)
                                            if _var162:
                                                _var163 = 'data analyst'
                                                return _var163
                                            else:
                                                _var164 = 'research'
                                                _var165 = words.lower()
                                                _var166 = (_var164 in _var165)
                                                _var167 = 'associate'
                                                _var168 = words.lower()
                                                _var169 = (_var167 in _var168)
                                                _var170 = (_var166 and _var169)
                                                if _var170:
                                                    _var171 = 'data analyst'
                                                    return _var171
                                                else:
                                                    _var172 = 'scientist'
                                                    _var173 = words.lower()
                                                    _var174 = (_var172 in _var173)
                                                    _var175 = 'research'
                                                    _var176 = words.lower()
                                                    _var177 = (_var175 in _var176)
                                                    _var178 = (_var174 and _var177)
                                                    if _var178:
                                                        _var179 = 'data analyst'
                                                        return _var179
                                                    else:
                                                        _var180 = 'statistical'
                                                        _var181 = words.lower()
                                                        _var182 = (_var180 in _var181)
                                                        _var183 = 'analyst'
                                                        _var184 = words.lower()
                                                        _var185 = (_var183 in _var184)
                                                        _var186 = (_var182 and _var185)
                                                        if _var186:
                                                            _var187 = 'data analyst'
                                                            return _var187
                                                        else:
                                                            _var188 = 'engineer'
                                                            _var189 = words.lower()
                                                            _var190 = (_var188 in _var189)
                                                            _var191 = 'data'
                                                            _var192 = words.lower()
                                                            _var193 = (_var191 in _var192)
                                                            _var194 = (_var190 and _var193)
                                                            if _var194:
                                                                _var195 = 'data engineer'
                                                                return _var195
                                                            else:
                                                                _var196 = 'engineer'
                                                                _var197 = words.lower()
                                                                _var198 = (_var196 in _var197)
                                                                _var199 = 'learning'
                                                                _var200 = words.lower()
                                                                _var201 = (_var199 in _var200)
                                                                _var202 = 'machine'
                                                                _var203 = words.lower()
                                                                _var204 = (_var202 in _var203)
                                                                _var205 = (_var198 and _var201 and _var204)
                                                                if _var205:
                                                                    _var206 = 'data engineer'
                                                                    return _var206
                                                                else:
                                                                    _var207 = 'statistical'
                                                                    _var208 = words.lower()
                                                                    _var209 = (_var207 in _var208)
                                                                    _var210 = 'programmer'
                                                                    _var211 = words.lower()
                                                                    _var212 = (_var210 in _var211)
                                                                    _var213 = (_var209 and _var212)
                                                                    if _var213:
                                                                        _var214 = 'data engineer'
                                                                        return _var214
                                                                    else:
                                                                        _var215 = 'statistical'
                                                                        _var216 = words.lower()
                                                                        _var217 = (_var215 in _var216)
                                                                        _var218 = 'statistician'
                                                                        _var219 = words.lower()
                                                                        _var220 = (_var218 in _var219)
                                                                        _var221 = (_var217 or _var220)
                                                                        if _var221:
                                                                            _var222 = 'statistician'
                                                                            return _var222
                                                                        else:
                                                                            _var223 = 'python'
                                                                            _var224 = words.lower()
                                                                            _var225 = (_var223 in _var224)
                                                                            _var226 = 'data'
                                                                            _var227 = words.lower()
                                                                            _var228 = (_var226 in _var227)
                                                                            _var229 = 'SQL'
                                                                            _var230 = words.lower()
                                                                            _var231 = (_var229 in _var230)
                                                                            _var232 = (_var225 or _var228 or _var231)
                                                                            if _var232:
                                                                                _var233 = 'data other'
                                                                                return _var233
                                                                            else:
                                                                                _var234 = 'scientist'
                                                                                _var235 = words.lower()
                                                                                _var236 = (_var234 in _var235)
                                                                                _var237 = 'science'
                                                                                _var238 = words.lower()
                                                                                _var239 = (_var237 in _var238)
                                                                                _var240 = (_var236 or _var239)
                                                                                if _var240:
                                                                                    _var241 = 'scientist'
                                                                                    return _var241
                                                                                else:
                                                                                    _var242 = 'misc'
                                                                                    return _var242
_var243 = 'Title_New'
_var244 = 'title'
_var245 = df_14[_var244]
_var246 = _var245.map(title_category)
df_15 = set_index_wrapper(df_14, _var243, _var246)
"\ndf['Title_New'].value_counts().nunique();\n\ndf['Title_New'].value_counts()"
_var247 = df_15.Title_New
_var247.value_counts()
_var248 = ['Title_New', 'city', 'state', 'company', 'ann_sal']
wdf = df_15[_var248]
_var249 = True
_var250 = True
wdf.reset_index(inplace=_var249, drop=_var250)
wdf.head()
_var251 = 'high_sal'
_var252 = 'ann_sal'
_var253 = wdf[_var252]

def _func1(x_0):
    _var254 = global_wrapper(wdf)
    _var255 = 'ann_sal'
    _var256 = _var254[_var255]
    _var257 = _var256.mean()
    _var258 = (x_0 > _var257)
    _var259 = 1
    _var260 = 0
    _var261 = (_var259 if _var258 else _var260)
    return _var261
_var262 = _var253.apply(_func1)
wdf_0 = set_index_wrapper(wdf, _var251, _var262)
wdf_0.head()
_var263 = '~ C(city) + C(state) + C(company)+ C(Title_New)'
X = patsy.dmatrix(_var263, wdf_0)
_var264 = 'high_sal'
_var265 = wdf_0[_var264]
y = _var265.values
_var266 = X.design_info
_var266.column_names
_var271 = 0.33
_var272 = 2
(_var267, _var268, _var269, _var270) = train_test_split(X, y, test_size=_var271, random_state=_var272)
X_train = _var267
X_test = _var268
Y_train = _var269
Y_test = _var270
_var273 = 'liblinear'
lr = LogisticRegression(solver=_var273)
lr_model = lr.fit(X_train, Y_train)
lr_0 = lr_model
y_pred = lr_model.predict(X_test)
y_score = lr_model.decision_function(X_test)
_var274 = [1, 0]
_var275 = confusion_matrix(Y_test, y_pred, labels=_var274)
conmat = np.array(_var275)
_var276 = ['over_mean', 'under_mean']
_var277 = ['predicted_overmean', 'predicted_undermean']
confusion = pd.DataFrame(conmat, index=_var276, columns=_var277)
print(confusion)
_var278 = classification_report(Y_test, y_pred)
print(_var278)
roc_auc_score(Y_test, y_score)
FPR = dict()
TPR = dict()
ROC_AUC = dict()
(_var279, _var280, _var281) = roc_curve(Y_test, y_score)
_var282 = 1
FPR_0 = set_index_wrapper(FPR, _var282, _var279)
_var283 = 1
TPR_0 = set_index_wrapper(TPR, _var283, _var280)
_ = _var281
_var284 = 1
_var285 = 1
_var286 = FPR_0[_var285]
_var287 = 1
_var288 = TPR_0[_var287]
_var289 = auc(_var286, _var288)
ROC_AUC_0 = set_index_wrapper(ROC_AUC, _var284, _var289)
_var290 = [11, 9]
plt.figure(figsize=_var290)
_var291 = 1
_var292 = FPR_0[_var291]
_var293 = 1
_var294 = TPR_0[_var293]
_var295 = 'ROC curve (area = %0.2f)'
_var296 = 1
_var297 = ROC_AUC_0[_var296]
_var298 = (_var295 % _var297)
_var299 = 4
plt.plot(_var292, _var294, label=_var298, linewidth=_var299)
_var300 = [0, 1]
_var301 = [0, 1]
_var302 = 'k--'
_var303 = 4
plt.plot(_var300, _var301, _var302, linewidth=_var303)
_var304 = [0.0, 1.0]
plt.xlim(_var304)
_var305 = [0.0, 1.05]
plt.ylim(_var305)
_var306 = 'False Positive Rate'
_var307 = 18
plt.xlabel(_var306, fontsize=_var307)
_var308 = 'True Positive Rate'
_var309 = 18
plt.ylabel(_var308, fontsize=_var309)
_var310 = 'Receiver operating characteristic for high/low income'
_var311 = 18
plt.title(_var310, fontsize=_var311)
_var312 = 'lower right'
plt.legend(loc=_var312)
plt.show()
C_vals = [0.0001, 0.001, 0.01, 0.1, 0.15, 0.25, 0.275, 0.33, 0.5, 0.66, 0.75, 1.0, 2.5, 5.0, 10.0, 100.0, 1000.0]
penalties = ['l1', 'l2']
_var313 = {'penalty': penalties, 'C': C_vals}
_var314 = False
_var315 = 15
gs = GridSearchCV(lr_0, _var313, verbose=_var314, cv=_var315)
gs_0 = gs.fit(X, y)
gs_0.best_params_
_var316 = gs_0.best_params_
_var317 = 'C'
_var318 = _var316[_var317]
_var319 = gs_0.best_params_
_var320 = 'penalty'
_var321 = _var319[_var320]
logreg = LogisticRegression(C=_var318, penalty=_var321)
cv_model = logreg.fit(X_train, Y_train)
logreg_0 = cv_model
cv_pred = cv_model.predict(X_test)
_var322 = logreg_0.classes_
cm = confusion_matrix(Y_test, cv_pred, labels=_var322)
_var323 = logreg_0.classes_
_var324 = logreg_0.classes_
cm_0 = pd.DataFrame(cm, columns=_var323, index=_var324)
cm_0
_var325 = logreg_0.classes_
_var326 = classification_report(Y_test, cv_pred, labels=_var325)
print(_var326)
'\nFPR = dict()\nTPR = dict()\nROC_AUC = dict()\n\n# For class 1, find the area under the curve\nFPR[1], TPR[1], _ = roc_curve(Y_test, cv_pred)\nROC_AUC[1] = auc(FPR[1], TPR[1])\n\n# Plot of a ROC curve for class 1 (has_cancer)\nplt.figure(figsize=[11,9])\nplt.plot(FPR[1], TPR[1], label=\'ROC curve (area = %0.2f)\' % ROC_AUC[1], linewidth=4)\nplt.plot([0, 1], [0, 1], \'k--\', linewidth=4)\nplt.xlim([0.0, 1.0])\nplt.ylim([0.0, 1.05])\nplt.xlabel(\'False Positive Rate\', fontsize=18)\nplt.ylabel(\'True Positive Rate\', fontsize=18)\nplt.title(\'Receiver operating characteristic for high/low income\', fontsize=18)\nplt.legend(loc="lower right")\nplt.show()'
