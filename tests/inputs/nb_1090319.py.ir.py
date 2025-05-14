

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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from patsy import dmatrices
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn import metrics
import seaborn as sns
_var0 = plt.style
_var1 = 'fivethirtyeight'
_var0.use(_var1)
import patsy
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.grid_search import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
_var2 = get_ipython()
_var3 = 'matplotlib'
_var4 = 'inline'
_var2.run_line_magic(_var3, _var4)
_var5 = get_ipython()
_var6 = 'load_ext'
_var7 = 'sql'
_var5.run_line_magic(_var6, _var7)
from sqlalchemy import create_engine
_var8 = 'postgresql://dsi_student:gastudents@dsi.c20gkj5cvu3l.us-east-1.rds.amazonaws.com/titanic'
engine = create_engine(_var8)
_var9 = 'SELECT * FROM train'
df = pd.read_sql(_var9, engine)
df.head()
df.describe()
df.info()
_var10 = df.corr()
_var11 = True
sns.heatmap(_var10, annot=_var11)
_var12 = 1
sns.set(font_scale=_var12)
_var13 = df.corr()
_var14 = True
sns.clustermap(_var13, annot=_var14)
_var15 = 1.5
sns.set(font_scale=_var15)
_var16 = ['Survived']
_var17 = ['Age', 'Parch', 'Pclass']
_var18 = 6
sns.pairplot(df, x_vars=_var16, y_vars=_var17, size=_var18)
_var19 = df.dropna()
sns.pairplot(_var19)
_var20 = 'AgeRounded'
_var21 = [round(x) for x in df['Age']]
df_0 = set_index_wrapper(df, _var20, _var21)
_var22 = 'AgeRounded'
_var23 = df_0[_var22]
_var23.unique()
_var24 = ['AgeRounded', 'Survived']
_var25 = df_0[_var24]
_var26 = ['AgeRounded']
_var27 = False
_var28 = _var25.groupby(_var26, as_index=_var27)
survived_age = _var28.mean()
survived_age
_var29 = 2.2
sns.set(font_scale=_var29)
_var30 = 45
_var31 = 30
_var32 = (_var30, _var31)
plt.figure(figsize=_var32)
_var33 = 'AgeRounded'
_var34 = 'Survived'
sns.barplot(x=_var33, y=_var34, data=survived_age)
df_0.head()
_var35 = 'female'
_var36 = [(1 if (x == 'female') else 0) for x in df['Sex']]
df_1 = set_index_wrapper(df_0, _var35, _var36)
df_1.head()
_var37 = ['Age']
df_2 = df_1.dropna(subset=_var37)
'according to National Statistical Standards the age distribution of the population for demographic purposes should be\ngiven in five-year age groups extending to 85 years and over.'
bins = [0, 4, 9, 14, 19, 24, 29, 34, 39, 44, 49, 54, 59, 64, 69, 74, 79, 84]
group_names = ['0-4 years', '5-9 years', '10-14 years', '15-19 years', '20-24 years', '25-29 years', '30-34 years', '35-39 years', '40-44 years', '45-49 years', '50-54 years', '55-59 years', '60-64 years', '65-69 years', '70-74 years', '75-79 years', '80-84 years']
_var38 = 'agebin'
_var39 = 'Age'
_var40 = df_2[_var39]
_var41 = pd.cut(_var40, bins, labels=group_names)
df_3 = set_index_wrapper(df_2, _var38, _var41)
_var42 = ['agebin', 'Sex', 'Pclass', 'Parch', 'Fare']
features = df_3[_var42]
features.head()
_var43 = 'Fare'
_var44 = df_3[_var43]
farelist = _var44.unique()
sorted(farelist)
_var45 = '~ C(agebin) + C(Sex) + C(Pclass)+ C(Parch)'
X = patsy.dmatrix(_var45, df_3)
X
_var46 = 'Survived'
_var47 = df_3[_var46]
y = _var47.values
_var52 = 0.33
_var53 = 66
(_var48, _var49, _var50, _var51) = train_test_split(X, y, test_size=_var52, random_state=_var53)
X_train = _var48
X_test = _var49
Y_train = _var50
Y_test = _var51
_var54 = 'liblinear'
lr = LogisticRegression(solver=_var54)
lr_model = lr.fit(X_train, Y_train)
lr_0 = lr_model
_var55 = lr_model.coef_
_var56 = 0
_var55[_var56]
_var57 = X.design_info
featurenames = _var57.column_names
_var58 = lr_model.coef_
_var59 = 0
_var60 = _var58[_var59]
_var61 = zip(featurenames, _var60)
_var62 = list(_var61)
coeff_logreg = pd.DataFrame(_var62)
_var63 = ['Feature_Name', 'Coefficient']
coeff_logreg_0 = set_field_wrapper(coeff_logreg, 'columns', _var63)
coeff_logreg_0
y_pred = lr_model.predict(X_test)
lr_model.predict_proba(X_test)
lr_model.score(X_test, Y_test)
_var64 = 3
_var65 = 'f1_weighted'
_var66 = cross_val_score(lr_model, X_test, Y_test, cv=_var64, scoring=_var65)
_var66.mean()
_var67 = lr_model.classes_
_var68 = classification_report(Y_test, y_pred, labels=_var67)
print(_var68)
_var69 = lr_model.classes_
cm = confusion_matrix(Y_test, y_pred, labels=_var69)
_var70 = lr_model.classes_
_var71 = lr_model.classes_
cm_0 = pd.DataFrame(cm, columns=_var70, index=_var71)
cm_0
y_score = lr_model.decision_function(X_test)
FPR = dict()
TPR = dict()
ROC_AUC = dict()
(_var72, _var73, _var74) = roc_curve(Y_test, y_score)
_var75 = 1
FPR_0 = set_index_wrapper(FPR, _var75, _var72)
_var76 = 1
TPR_0 = set_index_wrapper(TPR, _var76, _var73)
_ = _var74
_var77 = 1
_var78 = 1
_var79 = FPR_0[_var78]
_var80 = 1
_var81 = TPR_0[_var80]
_var82 = auc(_var79, _var81)
ROC_AUC_0 = set_index_wrapper(ROC_AUC, _var77, _var82)
_var83 = [11, 9]
plt.figure(figsize=_var83)
_var84 = 1
_var85 = FPR_0[_var84]
_var86 = 1
_var87 = TPR_0[_var86]
_var88 = 'ROC curve (area = %0.2f)'
_var89 = 1
_var90 = ROC_AUC_0[_var89]
_var91 = (_var88 % _var90)
_var92 = 4
plt.plot(_var85, _var87, label=_var91, linewidth=_var92)
_var93 = [0, 1]
_var94 = [0, 1]
_var95 = 'k--'
_var96 = 4
plt.plot(_var93, _var94, _var95, linewidth=_var96)
_var97 = [0.0, 1.0]
plt.xlim(_var97)
_var98 = [0.0, 1.05]
plt.ylim(_var98)
_var99 = 'False Positive Rate'
_var100 = 18
plt.xlabel(_var99, fontsize=_var100)
_var101 = 'True Positive Rate'
_var102 = 18
plt.ylabel(_var101, fontsize=_var102)
_var103 = 'Receiver operating characteristic for high/low income'
_var104 = 18
plt.title(_var103, fontsize=_var104)
_var105 = 'lower right'
plt.legend(loc=_var105)
plt.show()
_var106 = ['l1', 'l2']
_var107 = (- 5)
_var108 = 1
_var109 = 50
_var110 = np.logspace(_var107, _var108, _var109)
_var111 = ['liblinear']
logreg_parameters = {'penalty': _var106, 'C': _var110, 'solver': _var111}
_var112 = 'penalty'
_var113 = logreg_parameters[_var112]
_var114 = 'C'
_var115 = logreg_parameters[_var114]
_var116 = {'penalty': _var113, 'C': _var115}
_var117 = False
_var118 = 15
gs = GridSearchCV(lr_model, _var116, verbose=_var117, cv=_var118)
gs_0 = gs.fit(X_train, Y_train)
gs_0.best_params_
gs_0.best_score_
_var119 = gs_0.best_params_
_var120 = 'C'
_var121 = _var119[_var120]
_var122 = gs_0.best_params_
_var123 = 'penalty'
_var124 = _var122[_var123]
logreg = LogisticRegression(C=_var121, penalty=_var124)
gs_model = logreg.fit(X_train, Y_train)
logreg_0 = gs_model
gs_pred = gs_model.predict(X_test)
_var125 = logreg_0.classes_
cm1 = confusion_matrix(Y_test, gs_pred, labels=_var125)
_var126 = logreg_0.classes_
_var127 = logreg_0.classes_
cm1_0 = pd.DataFrame(cm1, columns=_var126, index=_var127)
cm1_0
y_score2 = gs_model.decision_function(X_test)
FPR_GS = dict()
TPR_GS = dict()
ROC_AUC_GS = dict()
(_var128, _var129, _var130) = roc_curve(Y_test, y_score2)
_var131 = 1
FPR_GS_0 = set_index_wrapper(FPR_GS, _var131, _var128)
_var132 = 1
TPR_GS_0 = set_index_wrapper(TPR_GS, _var132, _var129)
__0 = _var130
_var133 = 1
_var134 = 1
_var135 = FPR_GS_0[_var134]
_var136 = 1
_var137 = TPR_GS_0[_var136]
_var138 = auc(_var135, _var137)
ROC_AUC_GS_0 = set_index_wrapper(ROC_AUC_GS, _var133, _var138)
_var139 = [11, 9]
plt.figure(figsize=_var139)
_var140 = 1
_var141 = FPR_GS_0[_var140]
_var142 = 1
_var143 = TPR_GS_0[_var142]
_var144 = 'ROC curve (area = %0.2f)'
_var145 = 1
_var146 = ROC_AUC_GS_0[_var145]
_var147 = (_var144 % _var146)
_var148 = 4
plt.plot(_var141, _var143, label=_var147, linewidth=_var148)
_var149 = 1
_var150 = FPR_0[_var149]
_var151 = 1
_var152 = TPR_0[_var151]
_var153 = 'ROC curve (area = %0.2f)'
_var154 = 1
_var155 = ROC_AUC_0[_var154]
_var156 = (_var153 % _var155)
_var157 = 4
plt.plot(_var150, _var152, label=_var156, linewidth=_var157)
_var158 = [0, 1]
_var159 = [0, 1]
_var160 = 'k--'
_var161 = 4
plt.plot(_var158, _var159, _var160, linewidth=_var161)
_var162 = [0.0, 1.0]
plt.xlim(_var162)
_var163 = [0.0, 1.05]
plt.ylim(_var163)
_var164 = 'False Positive Rate'
_var165 = 18
plt.xlabel(_var164, fontsize=_var165)
_var166 = 'True Positive Rate'
_var167 = 18
plt.ylabel(_var166, fontsize=_var167)
_var168 = 'Receiver operating characteristic for high/low income'
_var169 = 18
plt.title(_var168, fontsize=_var169)
_var170 = 'lower right'
plt.legend(loc=_var170)
plt.show()
knn = KNeighborsClassifier()
_var171 = 1
_var172 = 51
_var173 = range(_var171, _var172)
_var174 = list(_var173)
_var175 = ['uniform', 'distance']
param_dict = dict(n_neighbors=_var174, weights=_var175)
_var176 = 'accuracy'
gscv = GridSearchCV(knn, param_dict, scoring=_var176)
gscv_model = gscv.fit(X_train, Y_train)
gscv_0 = gscv_model
_var177 = gscv_model.best_estimator_
_var177.get_params()
gscv_0.best_params_
gscv_0.best_score_
gscv_ypred = gscv_model.predict(X_test)
_var178 = classification_report(Y_test, gscv_ypred)
print(_var178)
cm2 = confusion_matrix(Y_test, gscv_ypred)
cm2_0 = pd.DataFrame(cm2)
cm2_0
