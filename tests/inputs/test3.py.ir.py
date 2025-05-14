

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
_var0 = 'data.csv'
df = pd.read_csv(_var0)
l = len(df)
_var1 = 'Survived'
y = df[_var1]
_var2 = 'Survived'
_var3 = 1
df_0 = df.drop(_var2, axis=_var3)
from sklearn import preprocessing
df1 = preprocessing.scale(df_0)
from sklearn.model_selection import train_test_split
_var8 = 0.33
_var9 = 42
(_var4, _var5, _var6, _var7) = train_test_split(df_0, y, test_size=_var8, random_state=_var9)
X_train = _var4
X_test = _var5
y_train = _var6
y_test = _var7
from sklearn.linear_model import LogisticRegression
_var10 = 'l2'
_var11 = 'sag'
_var12 = 0
clf = LogisticRegression(penalty=_var10, solver=_var11, random_state=_var12)
clf_0 = clf.fit(X_train, y_train)
y_pred = clf_0.predict(X_test)
